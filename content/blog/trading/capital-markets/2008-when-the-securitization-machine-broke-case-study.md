---
title: "2008: When the Securitization Machine Broke — A Capital-Markets Case Study"
date: "2026-06-21"
publishDate: "2026-06-21"
description: "How the chain from a home loan to a AAA bond stretched until nobody checked the risk, why a default shock made collateral unvaluable, and how the run that followed froze primary issuance overnight."
tags: ["capital-markets", "securitization", "2008-crisis", "subprime", "shadow-banking", "abcp", "repo", "credit-default-swaps", "cdo", "central-clearing"]
category: "trading"
subcategory: "Capital Markets"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — The 2008 crisis was, at its core, a failure of *trust in whether a security could be valued and sold* — and when that trust vanished, the primary market for securitized product died overnight.
>
> - The credit chain stretched from borrower to broker to originator to SPV to CDO to investor until **nobody owned the credit risk or checked it** — the "originate-to-distribute" model.
> - Rating agencies stamped **AAA** on subprime CDOs using a **flawed low-correlation assumption**; in a national housing bust, losses arrived together and the "safe" senior tranche was wiped.
> - The funding side was a **shadow bank**: maturity-mismatched conduits and SIVs borrowing overnight (ABCP, repo) to hold long CDOs. When collateral became unvaluable, lenders refused to roll, repo haircuts spiked, and a money fund **broke the buck**.
> - AIG had written **uncollateralised, uncleared CDS** on \$400bn+ of CDOs with no capital behind it — exactly the bilateral-derivative hole a central clearinghouse exists to close.
> - The one number to remember: US non-agency securitization issuance collapsed from roughly **\$700bn in 2007 to \$180bn in 2008** — the machine seized when its securities became unsellable.

## A morning when nothing would trade

On the morning of 9 August 2007, BNP Paribas froze three of its investment funds. The reason it gave was quietly devastating: "The complete evaporation of liquidity in certain market segments… has made it impossible to value certain assets fairly regardless of their quality or credit rating." Translated out of bank-speak: *we own these securities, but we no longer know what they are worth, because there is no one to sell them to.*

That sentence is the whole crisis in miniature. A capital market is a machine that turns savings into long-term investment, and it runs on a deal nobody signs but everyone relies on: I will fund a thirty-year mortgage, or buy a complicated bond backed by ten thousand of them, *only because I believe I can sell my claim tomorrow morning at a price I can roughly guess today*. Secondary-market liquidity is what makes primary issuance possible. The securitization machine — the assembly line that turns illiquid loans into tradable bonds — is the most spectacular expression of that idea ever built. And in 2007 and 2008, it ran in reverse: the moment buyers could no longer value the securities, they stopped trading them; the moment they stopped trading them, no new ones could be issued; and the funding that the whole edifice rested on disappeared in weeks.

This post is the series' spine stated in the negative. Everywhere else we argue that trust and liquidity in the secondary market are what let the primary market raise capital. Here we watch what happens when that trust breaks. We will follow the chain in order — the setup, the ratings failure, the run on the shadow bank, the AIG derivatives hole, the seize, and the reforms — and at every step the failure will turn out to be the *same* failure: a security people could no longer value, and therefore no longer sell.

![How the securitization machine broke in 2008 shown as a pipeline](/imgs/blogs/2008-when-the-securitization-machine-broke-case-study-1.png)

## Foundations: how the securitization machine is supposed to work

Before we can watch it break, we need the machine in working order. Strip away the acronyms and securitization is a simple idea about plumbing for money.

Start with an everyday picture. A bank lends you \$300,000 to buy a house. That loan is an *asset* to the bank — a promise of monthly payments for thirty years — but it is a deeply inconvenient asset. It is illiquid (you cannot easily sell one mortgage), it ties up the bank's capital for decades, and it concentrates risk in one borrower. **Securitization** is the trick that fixes all three at once: pool thousands of these loans together, place them in a separate legal box, and sell *bonds* backed by the pool's cash flows to investors all over the world. The loans become a security; the security trades; the bank gets its money back to lend again.

Let me define the cast of characters, because the whole story is about how this chain gets too long.

- **Borrower** — the household taking the mortgage.
- **Broker / originator** — the lender who writes the loan. Originally a bank that kept the loan and therefore cared whether you could repay.
- **SPV (special-purpose vehicle)** — the "separate legal box," also called a trust or conduit. It buys the pool of loans and issues bonds against them. It is bankruptcy-remote: if the originator fails, the SPV's assets are walled off.
- **Tranches** — the SPV does not issue one kind of bond; it slices the pool's cash flows into layers of different risk. The **senior** tranche gets paid first and absorbs losses last; the **equity** (or "first-loss") tranche gets paid last and eats the first losses. In between sits the **mezzanine**.
- **CDO (collateralised debt obligation)** — a securitization *of securitizations*. Take the leftover mezzanine tranches from many mortgage deals, pool *those*, and re-tranche them. A CDO is a bond backed by other bonds.
- **Investor** — pension funds, money-market funds, foreign banks, insurers: the savers at the end of the chain whose money funds the houses at the beginning.

There is one more concept the rest of the post leans on: the difference between a **liquidity** problem and a **solvency** problem. A firm is *insolvent* when its assets are genuinely worth less than its debts — it has lost money and cannot pay. A firm is *illiquid* when its assets are worth more than its debts but it cannot turn them into cash fast enough to meet a payment coming due today. In calm markets the two are easy to tell apart. In a panic they blur, because the only way to prove your assets are worth what you claim is to sell some — and selling into a frozen market crashes the price, which can turn a liquidity problem into a solvency one in an afternoon. Most of 2008's institutional failures began as liquidity crises and were *converted* into solvency crises by forced selling. Keeping that distinction in mind is the difference between understanding the crisis and just narrating it.

The genius of tranching is that it manufactures safety out of risk. If the pool of risky subprime loans is sliced so that the bottom 20% absorbs all the early losses, the top 80% looks very safe — it only takes a hit if more than a fifth of the entire pool defaults. That is the logic agencies used to stamp the senior tranche **AAA**, the same rating as a US Treasury bond. Hold that idea; it is the load-bearing assumption that snapped.

If you want the mechanics built from scratch — how a pool becomes a bond, why the SPV is bankruptcy-remote, how the waterfall pays out — read the sibling [securitization from first principles: turning loans into bonds](/blog/trading/capital-markets/securitization-from-first-principles-turning-loans-into-bonds) and the banking-side companion [securitization: how banks turn loans into securities](/blog/trading/banking/securitization-how-banks-turn-loans-into-securities). For the recursive tranche-of-tranches structure, see [CDOs, CLOs, and the tranching of tranches](/blog/trading/capital-markets/cdos-clos-and-the-tranching-of-tranches). This post takes those mechanics as read and asks the harder question: what makes the whole machine *trustworthy enough to run*, and what happens when the trust goes.

## Step 1 — The setup: cheap money, a housing boom, and a chain nobody policed

Three things lined up in the early 2000s to overload the machine.

First, **cheap money**. After the dot-com bust and 9/11, the Federal Reserve cut its policy rate to 1% by 2003 and held it low. Savers worldwide — including enormous pools of reserves from exporting nations — hunted for yield. A AAA-rated bond paying more than a Treasury was exactly what they wanted, and the securitization machine could manufacture it on demand.

Second, a **housing boom**. US home prices rose every year from the mid-1990s through 2006. Rising prices made mortgage lending look almost riskless: if a borrower stopped paying, the house could be sold for more than the loan. This belief — that prices only go up — is the single assumption underneath everything that followed.

Third, and most corrosively, the **originate-to-distribute model** rewired who cared about credit quality. In the old world, the bank that wrote your loan kept it, so it screened you carefully — its own money was at stake. In the new world, the originator wrote the loan, sold it within weeks to an SPV, and pocketed a fee. The SPV bundled it into a bond and sold that to a CDO desk, which re-tranched it and sold *that* to an investor. At every handoff, the credit risk moved one step further from the person who had actually met the borrower. By the time the risk reached the investor, it had passed through five sets of hands, none of whom expected to hold it.

#### Worked example: how the fee chain breaks the incentive to screen

Consider a single \$200,000 subprime mortgage. The mortgage broker earns a \$4,000 origination fee (2%) the day it closes — and earns *nothing* for turning the borrower down. The originator who funds it sells the loan to an SPV for \$202,000, booking a \$2,000 gain, and recycles the cash into the next loan. The SPV's arranger earns roughly 0.5–1.5% structuring the bonds — call it \$1,500 on this loan's slice. The CDO desk earns another structuring and management fee. Add it up: roughly **\$8,000–\$10,000 of fees were generated by originating and packaging a single \$200,000 loan**, and *every dollar of that was earned at origination, none of it contingent on the borrower actually repaying*. When the people who decide whether to lend get paid for volume and bear none of the default, screening is not a little weaker — it is structurally pointed the wrong way. That is why "NINJA" loans (no income, no job, no assets) could exist: nobody in the chain who approved them planned to own the risk.

It is worth pausing on *why* the demand side was so insatiable, because that is what pulled the whole chain taut. Through the 2000s, a wall of foreign savings — China's accumulating reserves, oil-exporter surpluses, conservative European banks — sought a home in dollar assets that were both safe and paid more than Treasuries. There is a finite supply of genuinely safe dollar bonds, so when global demand for "safe" outstrips that supply, the market does what markets do: it *manufactures* the missing product. The securitization machine existed to convert a near-unlimited supply of risky raw material (subprime mortgages) into the scarce, high-demand product (AAA-rated bonds). The flawed correlation assumption was not a side error; it was the necessary fiction that let the manufacturing happen at all. Demand for AAA created the supply of AAA, whether or not the underlying loans deserved it.

That feedback loop is what made the boom self-reinforcing. More AAA demand meant higher prices for securitized bonds, which meant originators could sell loans for more, which meant they wrote more loans, which they could only do by relaxing standards once the creditworthy borrowers ran out. Falling standards were not a bug in the boom — they were the boom's logical endpoint, the only way to keep feeding a machine whose appetite had outgrown its supply of good borrowers.

The chain had stretched until the credit risk was an orphan. And the volume it produced was staggering.

![US subprime mortgage origination by year](/imgs/blogs/2008-when-the-securitization-machine-broke-case-study-2.png)

Subprime origination roughly tripled from about \$190bn in 2001 to a peak near \$625bn in 2005, then stayed enormous through 2006 — before collapsing to \$191bn in 2007 and a rounding error by 2008. The shape of that chart *is* the crisis: a vertical run-up fed by a machine that no longer screened, followed by a cliff when the machine seized. Tie it back to the spine: each of those loans was raised in the *primary* market only because the originator was certain it could be *sold* into the securitization pipeline tomorrow. Primary issuance was riding entirely on secondary-market appetite.

## Step 2 — The ratings failure: AAA from BBB, and a correlation that wasn't there

Here is the alchemy at the heart of the machine. A subprime mortgage pool is full of risky loans — individually, none of them deserves a top rating. Yet the senior tranche of a CDO built from them was routinely rated AAA. How?

The answer is tranching plus an assumption about **correlation**. Correlation is the degree to which defaults happen *together*. If defaults are largely independent — one borrower in Florida losing his job has nothing to do with one in Nevada — then the law of large numbers works in your favour. In any given year a predictable, small fraction of the pool defaults, the first-loss tranche absorbs it, and the senior tranche is untouched. Under a low-correlation assumption, you can mathematically show that the top 80% of a pool of BBB-quality loans almost never takes a loss, and "almost never takes a loss" is what AAA *means*.

![The ratings model's fatal correlation assumption shown before and after](/imgs/blogs/2008-when-the-securitization-machine-broke-case-study-3.png)

The models the agencies used — famously a Gaussian copula calibrated on a short, benign history — plugged in a *low* default correlation. That history had never included a nationwide fall in house prices, because there hadn't been one since the 1930s. So the models assumed that Florida and Nevada and Ohio defaults were nearly independent. They were not. When home prices fell *everywhere at once* in 2007–2008, defaults stopped being independent and became one giant correlated event. The diversification that justified the AAA stamp evaporated exactly when it was needed.

There was a second corrosion: the agencies were paid by the issuers whose bonds they rated. This **issuer-pays** model meant the agency that gave the friendliest assumptions won the business — a conflict of interest baked into the plumbing. We treat the ratings mechanics and the agency business model in depth in [CDOs, CLOs, and the tranching of tranches](/blog/trading/capital-markets/cdos-clos-and-the-tranching-of-tranches); here the point is the consequence: a rating is a *claim that a security can be trusted*, and when the claim rested on a correlation that didn't hold, the trust was hollow.

![Tranche stack of a 100-unit subprime CDO](/imgs/blogs/2008-when-the-securitization-machine-broke-case-study-5.png)

#### Worked example: how a 10% pool loss wipes a "AAA-from-BBB" senior tranche

Take a \$100 CDO pool tranched the way the chart shows: a 5% equity tranche, a 15% mezzanine tranche, and an 80% senior tranche rated AAA. Losses hit from the bottom up.

- A **5% pool loss** (\$5) is fully absorbed by the equity tranche. Equity holders are wiped; mezzanine and senior are untouched. The AAA looks bulletproof.
- A **10% pool loss** (\$10) exhausts the \$5 equity tranche and eats \$5 of the \$15 mezzanine. Mezzanine takes a 33% hit; the AAA senior is *still* untouched. This is the regime the models assumed was the worst case.
- Now the correlated bust. Subprime pools didn't lose 10% — many lost **30–40%** as defaults arrived together and recovery values collapsed (a foreclosed house in a falling market fetches far less than the loan). A **25% pool loss** (\$25) wipes the entire \$5 equity *and* the entire \$15 mezzanine (\$20) and then takes \$5 out of the senior tranche. The "AAA" bond that was supposed to *almost never* lose a cent has now taken a **6.25% loss** (\$5 of \$80) — and the market, seeing losses blow through two layers, repriced it not at 94 cents but at 40, because nobody could be sure where the losses would stop.

The intuition: **tranching only converts risk into safety if losses arrive a few at a time; under high correlation they arrive all at once, and the "safe" senior tranche is simply the last domino, not an immune one.** The AAA was never a property of the bond — it was a property of an assumption, and the assumption failed.

## Step 3 — The run on the shadow bank

Now follow the *funding* side, because this is where a credit problem became a liquidity catastrophe.

A normal bank takes deposits (short-term money it owes you on demand) and makes loans (long-term assets). That maturity mismatch is the essence of banking, and it is dangerous: if all the depositors want their money at once, the bank cannot sell its thirty-year loans fast enough — a bank run. Society tamed that danger with deposit insurance and a central-bank lender of last resort.

The securitization machine rebuilt the same maturity mismatch *outside* that safety net. It is called the **shadow banking system**, and its workhorses were **conduits** and **SIVs** (structured investment vehicles): legal entities that held long-dated CDOs and mortgage bonds and funded them by issuing **asset-backed commercial paper (ABCP)** — IOUs that mature in 1 to 90 days — plus borrowing in the **repo** market. Repo (repurchase agreement) is a loan secured by a security: I sell you a bond today and agree to buy it back tomorrow for slightly more; the difference is your interest, and you protect yourself with a **haircut** (you only lend, say, 98 cents against a \$1 bond). The whole sibling on this plumbing — [securities lending and repo: the financing plumbing](/blog/trading/capital-markets/securities-lending-and-repo-the-financing-plumbing) — is worth reading alongside [covered bonds, ABCP, and the shadow funding chain](/blog/trading/capital-markets/covered-bonds-abcp-and-the-shadow-funding-chain).

The catch is that short-term funding has to be *rolled over* constantly. A conduit holding a 10-year CDO but funding it with 30-day paper has to find a new lender twelve-plus times a year. That works only as long as lenders trust the collateral. The instant they don't, the run begins.

![The run on the shadow bank shown as a pipeline](/imgs/blogs/2008-when-the-securitization-machine-broke-case-study-6.png)

Watch the dominoes. Subprime defaults rise. The value of the CDOs sitting in the conduits becomes *unknowable* — exactly the BNP Paribas sentence. A money-market fund that has been buying the conduit's ABCP looks at its holdings and says: I cannot value that, so I will not buy any more of it. The conduit cannot roll its paper. It turns to repo, but its repo lenders, seeing the same uncertainty, jack up the haircut. It is forced to sell assets into a market with no buyers, which pushes prices down further, which makes everyone's collateral worth even less — a fire-sale doom loop. And in September 2008 the **Reserve Primary Fund**, which held \$785m of Lehman Brothers paper, marked its share price below \$1.00 — it **"broke the buck"** — triggering a \$300bn run on money funds in days, because the one thing a money fund promises is that a dollar in is always a dollar out.

#### Worked example: a repo haircut going 2% to 25% and the deleveraging it forces

A shadow bank holds a \$50bn book of mortgage-backed securities, funded in repo. In good times the haircut is **2%**: lenders advance 98 cents per dollar of collateral, so the firm posts \$1bn of its own equity and borrows \$49bn. Leverage is 50-to-1 — beautiful in a boom.

Now collateral becomes suspect and lenders raise the haircut to **25%**. To hold \$50bn of securities, the firm must now fund 25% — **\$12.5bn** — with its own money, up from \$1bn. It does not have an extra \$11.5bn lying around. So it must *shrink the book*. With \$1bn of equity and a 25% haircut, the most it can finance is \$1bn ÷ 0.25 = **\$4bn of securities**. It has to dump \$46bn — 92% of the book — into a falling market. Multiply that across every shadow bank deleveraging at once and you have a forced, synchronized fire sale that drives prices below any fundamental value.

The intuition: **a haircut is the secured-lending version of a deposit run, and a 2%→25% move turns 50-to-1 leverage into a forced 92% liquidation overnight — not because the assets defaulted, but because nobody would finance them.** This is the link from a credit problem (some loans go bad) to a liquidity collapse (everything must be sold). For how the same mechanism nearly destroyed a single fund a decade earlier, see [LTCM 1998: when genius failed](/blog/trading/finance/ltcm-1998-when-genius-failed).

#### Worked example: the ABCP roll freeze on a conduit

Take a conduit, "Mercury Funding," holding \$20bn of mortgage CDOs and funding them entirely with 30-day ABCP, rolled in roughly \$700m daily tranches. For years the roll is automatic — a money fund whose old paper matures simply buys new paper. In August 2007 the money fund's risk committee can no longer value Mercury's CDOs and refuses to roll. On day one, \$700m of paper matures with no buyer. Mercury draws its bank backstop line. On day two, another \$700m matures; the backstop is finite. By the end of the month roughly \$20bn must be refinanced and cannot be. Mercury must either sell \$20bn of CDOs into a frozen market (at maybe 50 cents on the dollar, crystallising a \$10bn loss) or push the assets back onto the sponsoring bank's balance sheet — which is exactly what happened across the system, dragging banks' hidden shadow exposures into the open. The intuition: **a conduit is a bank with no deposit insurance and no central-bank backstop; when its overnight lenders blink, it has days, not months, and there is no fire exit that isn't a fire sale.** This is the failure analysed in [covered bonds, ABCP, and the shadow funding chain](/blog/trading/capital-markets/covered-bonds-abcp-and-the-shadow-funding-chain).

This was not theory. In September 2007, **Northern Rock**, a British mortgage lender that funded itself heavily in wholesale and securitization markets rather than deposits, faced exactly this roll freeze and suffered the first British bank-run queue-on-the-pavement in over a century — not because its loans had defaulted, but because the markets it borrowed from had shut. In March 2008, **Bear Stearns** — a \$400bn-balance-sheet investment bank funded substantially through overnight repo — lost the confidence of its repo lenders over a single week; once they refused to roll, Bear went from solvent-looking to sold-to-JPMorgan-for-\$2-a-share in days. Both were runs in the precise mechanical sense of this section: a maturity-mismatched borrower whose short-term funding evaporated faster than it could sell its long-term assets. Bear Stearns was the dress rehearsal; the market read the Fed's rescue of its counterparties as a signal that big dealers were too connected to fail — which made Lehman's failure six months later all the more shocking.

## Step 4 — AIG and the CDS hole

There was one more way to be exposed to subprime without owning a single mortgage bond: by *insuring* one. A **credit default swap (CDS)** is exactly that — a contract where one party pays a premium and the other promises to pay out if a named bond defaults. It is insurance on a bond, except it trades as a derivative and, crucially in 2008, was almost entirely **bilateral and uncleared**: two parties signed a contract directly, with no exchange and no central counterparty standing between them.

AIG Financial Products, a small London-based unit of the giant insurer, wrote CDS protection on more than **\$400bn** of securities, including roughly \$78bn referencing subprime CDOs. To AIG it looked like free money: collect premiums on bonds that "would never default" because they were AAA. The fatal feature was that AIG, with its own AAA credit rating, was allowed to write this protection with **almost no collateral posted up front and no capital reserved against it**. The contracts contained collateral-posting triggers: if the insured CDOs were downgraded or marked down, AIG had to post cash. As long as everyone trusted AIG, the triggers slept.

#### Worked example: the netting and collateral hole an uncleared CDS opens

Suppose AIG has written \$78bn of CDS on subprime CDOs and posted essentially zero collateral, because at inception the CDOs were AAA and AIG was AAA. Now the CDOs are marked down 20% and AIG itself is downgraded. The contracts' triggers fire and AIG suddenly owes **collateral calls of \$20bn+** to dozens of counterparties — Goldman, Société Générale, Deutsche Bank, others — *all at once*. AIG does not have \$20bn of spare cash. Each counterparty is owed in full by a firm that cannot pay, and because the trades are bilateral, **there is no netting across them and no shared default fund** — every bank faces AIG directly and alone. If AIG defaults, each counterparty eats its own loss and scrambles to replace the hedge in a market where everyone is doing the same thing simultaneously.

Now contrast the world a central counterparty would have created. A **CCP (central clearinghouse)** stands in the middle of every trade — it becomes buyer to every seller and seller to every buyer (this is *novation*). Three things change. First, it **demands initial margin** up front and **variation margin daily**, so AIG could never have built a \$400bn book on zero collateral — the margin would have forced it to reserve cash from day one. Second, it **nets** exposures: a member's offsetting longs and shorts collapse to a single net obligation, so the system-wide collateral demand is a fraction of the gross. Third, it **mutualises** the tail: a defaulter's margin and a shared guarantee fund absorb the loss in an orderly waterfall, instead of each counterparty being left to fend for itself. The intuition: **AIG's CDS book was a \$400bn promise with no money behind it and no one in the middle to demand any — precisely the bilateral-derivative hole a clearinghouse exists to close, which is why post-crisis reform made central clearing of standardised derivatives mandatory.** For the mechanics of novation, margin, and the default waterfall, read [the clearinghouse: how a CCP removes counterparty risk](/blog/trading/capital-markets/the-clearinghouse-how-a-ccp-removes-counterparty-risk).

There is a subtle point about *why* the CDS made things worse rather than spreading risk as intended. In principle, insurance disperses risk: if a hundred parties each insure a little of a bond, the loss is shared and no one is destroyed. The CDS market did the opposite — it *concentrated* risk, because the protection-sellers were a handful of large dealers and AIG, and it *hid* the concentration, because the contracts were private and bilateral. No regulator, and no counterparty, could see the total size of AIG's book until the collateral calls revealed it all at once. Worse, CDS could be written far in excess of the bonds they referenced (you can buy "insurance" on a house you do not own), so the notional amount of protection on subprime CDOs grew to a multiple of the actual bonds outstanding. When the reference bonds soured, the losses to be settled were larger than the bonds themselves. Opacity plus concentration plus leverage is the recipe for a single failure becoming everyone's failure — and it is exactly the recipe a clearinghouse, with its public position data, netting, and margin, is designed to defuse.

The US government ultimately committed up to \$182bn to AIG — not to save AIG itself, but to make whole the banks on the other side of those uncleared contracts, because letting them fail in a chain reaction was judged worse.

## Step 5 — The seize: Lehman, the freeze, and primary issuance to zero

By September 2008 every link we have described was straining at once. Subprime defaults were correlated and severe (Step 2). The shadow banks were being run on (Step 3). The derivatives that were supposed to spread risk had concentrated it in a single uncapitalised insurer (Step 4). The connective tissue that finally tore was the **interbank funding market** — the short-term lending banks do to each other to manage daily cash.

When **Lehman Brothers** filed for bankruptcy on 15 September 2008, the assumption that a big dealer bank could not be allowed to fail died with it. Lenders stopped trusting *everyone*, because no one could be sure who was exposed to whom through the opaque web of CDOs, repo, and uncleared CDS. The price banks charge each other for unsecured three-month money — the spread of LIBOR over the expected policy rate — spiked to levels never seen before. Repo against anything but Treasuries became almost impossible. The Reserve Primary Fund broke the buck the same week, and the run on money funds choked off the ABCP market that the conduits depended on. Every funding channel froze simultaneously.

And then the part that matters most for this series: **the primary market for securitized product went to zero.** You cannot issue a new bond that no one can value and no one will fund. New private-label mortgage securitization, which had been a roughly \$700bn-a-year business, simply stopped.

![US non-agency securitization issuance by year](/imgs/blogs/2008-when-the-securitization-machine-broke-case-study-4.png)

Read the chart against the spine. Non-agency securitization issuance fell from about \$700bn in 2007 to \$180bn in 2008 and \$150bn in 2009 — a roughly 75–80% collapse in a single year. This is not a story about prices wobbling. It is the *creation* engine of an entire capital-market segment shutting off, because the precondition for creating a security — that someone trusts it enough to buy and fund it tomorrow — had vanished. The secondary-market liquidity that makes primary issuance possible went to zero, and primary issuance went with it. The machine did not slow down; it stopped.

To see how central this securitized funding had become, look at where it sits in the broader issuance picture once markets healed.

![US debt issuance by type in 2023](/imgs/blogs/2008-when-the-securitization-machine-broke-case-study-7.png)

Even today — on a log scale, because Treasury issuance dwarfs everything — mortgage securitization (MBS) and asset-backed securities (ABS) together raise well over a trillion dollars a year of funding for households and consumers. That is the channel that froze. When it froze, the funding for new mortgages, car loans, and credit cards froze with it, which is how a problem in structured bonds reached into the real economy and became a recession.

Stopping the freeze required the public sector to step into the role private markets had abandoned: the buyer and lender of last resort. The Federal Reserve stood up an alphabet soup of emergency facilities — among them the Commercial Paper Funding Facility, which bought commercial paper directly when no private buyer would, and the Term Asset-Backed Securities Loan Facility (TALF), which lent against new ABS to restart the securitization market itself. The logic was explicit and is the mirror image of this whole post: because no private actor would buy a security it could not value, a backstop with a printing press and a long horizon had to become the buyer, restoring just enough confidence in price to let trading — and therefore issuance — resume. The Troubled Asset Relief Program (TARP) injected capital directly into banks for the same reason: not because the banks were necessarily insolvent, but because the market could not tell which ones were, and that uncertainty alone was enough to freeze everything. Trust, once gone, could only be manufactured by an actor whose word the market could not doubt.

## Common misconceptions

**"Subprime mortgages caused the crisis."** Subprime defaults were the *trigger*, not the mechanism. US subprime mortgages totalled on the order of \$1.3 trillion — a large number, but small against a \$14 trillion economy and global capital markets in the hundreds of trillions. A wave of subprime defaults alone would have caused painful losses, not a global seizure. What turned a credit loss into a systemic crisis was **leverage funded by runnable short-term money and a derivatives web that hid who owed whom** — the shadow-banking and CDS layers. The lesson is structural: it is rarely the bad asset that kills you; it is *how the bad asset was financed*.

**"The rating agencies were simply incompetent."** The models were wrong about correlation, but the deeper flaw was an incentive: the **issuer-pays** business model meant agencies competed by being generous. A AAA rating is a statement that a security can be trusted; when the firm issuing that statement is paid by the firm that benefits from a high rating, the statement is compromised before any model is run. The fix was not "smarter models" but changing the conflict — though issuer-pays largely survived the reforms, which is part of "what wasn't fixed."

**"AIG failed because it was an insurance company that took too much risk."** AIG's regulated insurance subsidiaries were largely fine. The hole was a derivatives desk writing **uncleared bilateral CDS with no margin and no capital** — exactly the activity that, run through a CCP with initial margin and a default fund, could never have grown that large unfunded. AIG is the canonical argument *for* central clearing, not against insurance.

**"Securitization is inherently toxic and should be banned."** Securitization is one of the most useful inventions in capital markets — it is how a regional bank can keep lending without running out of capital, and how a saver in Singapore can fund a mortgage in Texas. The post-crisis **CLO** (collateralised *loan* obligation) market, built on the same tranching idea but with retained risk and better underwriting, came through the 2020 COVID shock with very few senior-tranche losses. The 2008 failure was not tranching; it was tranching *plus* orphaned credit risk, *plus* a flawed correlation assumption, *plus* runnable funding. Fix those and the machine is genuinely valuable.

**"Everyone knew it was a bubble and got out in time."** Almost no one in the chain owned the risk knowingly. Money funds thought they held safe, liquid paper. Pension funds thought they held AAA bonds. Even the banks that built the CDOs retained "super-senior" slices they believed were riskless and warehoused unsold inventory when the music stopped. The defining feature of 2008 is not greed-with-foresight; it is a whole system trusting a rating that was wrong.

## How it shows up in real markets: the lessons and reforms

The crisis rewrote the rulebook. Each major reform maps directly onto one snapped link in the trust chain.

![What 2008 broke and what the reforms fixed shown as a matrix](/imgs/blogs/2008-when-the-securitization-machine-broke-case-study-8.png)

**Skin in the game (risk retention).** The Dodd-Frank Act and European rules require the originator or sponsor of a securitization to retain typically **5%** of the credit risk it creates — it cannot sell the whole thing and walk away. This directly attacks the originate-to-distribute orphaning of risk: if you must eat 5% of the losses, you screen the borrowers. It is the single cleanest fix, and it is why the modern CLO market underwrites far more carefully.

**Mandatory central clearing of standardised derivatives.** The G20 agreed in 2009 that standardised over-the-counter derivatives must be cleared through CCPs. This is the AIG fix: a clearinghouse demands initial and daily variation margin and nets exposures, so no one can build a \$400bn unfunded book again, and a single failure is absorbed by margin and a default fund rather than cascading bilaterally. The trade-off is that risk now concentrates *in* the CCPs, which become systemically critical — a point the sibling [the clearinghouse: how a CCP removes counterparty risk](/blog/trading/capital-markets/the-clearinghouse-how-a-ccp-removes-counterparty-risk) explores in full.

**Money-fund reform.** After Reserve Primary broke the buck, the US required institutional prime money funds to use a **floating net asset value** (so a dollar in is *not* guaranteed to be a dollar out, removing the run incentive of a fixed \$1.00) and added liquidity fees and gates. The aim is to make a money fund behave like what it is — a fund — rather than a deposit-like promise that invites a run. The 2020 COVID dash-for-cash showed this was only partly successful, prompting further tweaks; hence the matrix marks it "mostly," not "largely," fixed.

**Higher capital, liquidity buffers, and stress tests.** Basel III forced banks to hold more and better capital, plus a **liquidity coverage ratio** requiring a buffer of liquid assets against 30 days of outflows — a direct answer to the maturity-mismatch run. Stress tests now ask, in effect, "what happens if collateral becomes unvaluable again?"

#### Worked example: how 5% risk retention changes the originator's math

Return to that \$200,000 subprime loan and its ~\$8,000 of front-loaded fees. Under the old model, the originator's expected profit was \$8,000 with essentially zero downside, regardless of whether the borrower repaid — so the rational move was maximum volume, minimum screening. Now impose 5% risk retention on the securitization. The sponsor must hold 5% of the deal's first-loss exposure. On a \$1bn pool, that is **\$50m of equity-tranche risk it cannot sell** — the very slice that gets wiped first when underwriting is bad. If sloppy lending pushes pool losses from an expected 4% to a realised 25%, that retained \$50m is gone before any outside investor loses a cent. Suddenly the originator's expected profit is fee income *minus* the expected loss on its retained slice, and screening out the worst borrowers raises that number. The intuition: **risk retention does not ban bad lending; it re-prices it onto the lender's own book, which is the only thing that reliably makes a lender care.** That single change is why post-2010 securitizations look so different from 2006 vintages.

What *wasn't* fully fixed is worth naming honestly. The issuer-pays ratings model survived. The shadow-banking maturity mismatch was tamed inside banks but migrated to non-bank lenders, money funds, and other corners that the 2020 turmoil exposed again. And the deepest issue — that in a true panic *any* security can become temporarily unvaluable, freezing markets that were liquid the day before — is not a thing rules can abolish. It is the permanent fragility at the heart of the machine, which is why central banks now keep emergency facilities ready to be the buyer of last resort when private trust evaporates.

## The takeaway: trust is the load-bearing wall

Step back from the acronyms and 2008 delivers one lesson, and it is the spine of this entire series turned inside out.

A capital market converts savings into long-term investment only because a security *can be sold*. Every figure in this post is a picture of that single dependency breaking. The originate-to-distribute chain worked only while each buyer trusted the next would buy. The AAA rating was valuable only while people trusted it described the bond. The conduits could fund long assets with overnight money only while lenders trusted the collateral. AIG's CDS were worth writing only while counterparties trusted AIG could pay. And the primary market for securitized bonds existed only while there was a secondary market to sell them into. When the trust at any one of those links failed, it propagated to all of them — because they were all the *same* trust wearing different clothes. The moment securities could not be valued, they could not be sold; the moment they could not be sold, they could not be issued; and the creation engine of an entire market shut off in a single year.

That is why the reforms are not really about subprime or CDOs or any one instrument. They are about rebuilding the conditions under which a security can be trusted: skin in the game so someone vouches for the credit, central clearing so a derivative promise has money behind it, capital and liquidity buffers so a funding wobble doesn't force a fire sale, and a lender of last resort for the moments when private trust still fails. The CLO market that rose from the wreckage — same tranching idea, retained risk, sturdier underwriting — is proof that the machine itself was never the problem. The problem was running it on trust that had nothing underneath it.

The practical way to use this: whenever you meet a market that looks miraculously liquid and a security that looks miraculously safe, ask the 2008 questions. *Who actually owns the credit risk, and do they bear it? What is this thing funded with, and what happens if that funding refuses to roll? And on a bad morning, who is the buyer?* If the honest answers are "nobody," "overnight money," and "no one" — you are looking at the securitization machine of 2006, no matter what decade it is.

## Further reading & cross-links

- [Securitization from first principles: turning loans into bonds](/blog/trading/capital-markets/securitization-from-first-principles-turning-loans-into-bonds) — how a pool of loans becomes a tradable bond, built from zero.
- [CDOs, CLOs, and the tranching of tranches](/blog/trading/capital-markets/cdos-clos-and-the-tranching-of-tranches) — the recursive structures and the ratings model at the centre of this story.
- [Covered bonds, ABCP, and the shadow funding chain](/blog/trading/capital-markets/covered-bonds-abcp-and-the-shadow-funding-chain) — the conduits and commercial paper that funded the machine and ran.
- [Securities lending and repo: the financing plumbing](/blog/trading/capital-markets/securities-lending-and-repo-the-financing-plumbing) — repo, haircuts, and the leverage that fire-sold.
- [The clearinghouse: how a CCP removes counterparty risk](/blog/trading/capital-markets/the-clearinghouse-how-a-ccp-removes-counterparty-risk) — the novation, margin, and default-waterfall machinery that the AIG hole was missing.
- [LTCM 1998: when genius failed](/blog/trading/finance/ltcm-1998-when-genius-failed) — the same leverage-and-funding doom loop a decade earlier, in miniature.
- [Securitization: how banks turn loans into securities](/blog/trading/banking/securitization-how-banks-turn-loans-into-securities) — the commercial-bank side of the same plumbing.
