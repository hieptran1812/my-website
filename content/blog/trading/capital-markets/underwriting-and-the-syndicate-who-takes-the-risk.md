---
title: "Underwriting and the Syndicate: Who Takes the Risk?"
date: "2026-06-21"
publishDate: "2026-06-21"
description: "How an offering actually gets sold — the three underwriting modes, the syndicate that shares the risk, how the fee splits, and the greenshoe that quietly props up the price."
tags: ["capital-markets", "underwriting", "syndicate", "ipo", "greenshoe", "investment-banking", "primary-market", "gross-spread", "bookrunner", "stabilisation"]
category: "trading"
subcategory: "Capital Markets"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — Underwriting is the business of *standing between* an issuer that wants cash now and a market that may or may not show up to buy; the whole question of underwriting is **who eats the loss if the buyers don't come.**
>
> - There are three modes: **firm commitment** (the bank buys the entire deal at a fixed price and resells it, so the bank's own capital is at risk), **best efforts** (the bank is just an agent and takes no inventory risk), and the **bought deal** (a firm commitment priced and printed overnight, the highest-risk version of all).
> - No single bank carries a large deal alone. It builds a **syndicate** — a lead bookrunner, joint bookrunners, co-managers, and a selling group — to spread the risk, widen distribution, and divide the credit on the league tables.
> - The bank's pay is the **gross spread** (often ~7% of an IPO, less for a bond), and it splits roughly **20 / 20 / 60** into a management fee, an underwriting fee, and a selling concession — most of the money goes to whoever actually *places* the shares.
> - The **greenshoe** lets the syndicate sell 115% of the deal, run a 15% short, and either buy stock back if it sags (propping the price) or exercise an option to cover if it rallies. The one number to remember: on a classic IPO, the bank's underwriting fee is often only **~20% of a ~7% spread** — it is paid mostly for *placement*, and the real money it risks is the unsold inventory.

On the morning of 19 May 2012, Facebook was supposed to be the easiest IPO in a decade. Demand was enormous, the price had been pushed up to \$38, and the deal had been upsized at the last minute. Then the Nasdaq's opening systems choked, the stock barely held its offer price, and over the following days it slid into the low \$30s and then the high \$20s. The underwriters — a syndicate led by Morgan Stanley — found themselves doing something most retail investors never see: stepping into the open market and *buying Facebook stock* to keep the price from collapsing through the \$38 offer. They could do this because the deal had been sold with a built-in over-allotment, and because the lead bank had, in effect, promised to be the buyer of last resort. For a few days, an investment bank's trading desk was the only thing standing between a marquee IPO and a public humiliation.

That episode is the whole subject of this post compressed into one anecdote. Selling a new security is not like selling apples. The issuer wants a *guaranteed* amount of cash, on a *known* date, at a *known* price — but the demand that determines whether the deal works only reveals itself over the days the deal is being marketed and then traded. Somebody has to bridge that gap. Somebody has to say "we will hand you the money even if the buyers turn out to be thinner than we hoped." That somebody is the underwriter, and the price they charge for crossing that bridge — and the elaborate machinery they build to survive crossing it — is what this post is about.

This is a post about the **primary market** — the engine of a capital market that *creates* securities to raise money, as opposed to the secondary market that merely trades them afterward. (For the full IPO storyline — mandate, due diligence, the prospectus, the roadshow, pricing night — see [the IPO process end to end](/blog/trading/capital-markets/the-ipo-process-end-to-end-from-mandate-to-first-trade); for how the actual *price* gets discovered, see [bookbuilding and price discovery](/blog/trading/capital-markets/bookbuilding-and-price-discovery-how-the-ipo-price-is-set).) Our narrow question here is the one the Facebook morning made vivid: **who bears the risk of the offering, and how do they survive it?**

![The underwriting syndicate structure as a tree from issuer to buyers](/imgs/blogs/underwriting-and-the-syndicate-who-takes-the-risk-1.png)

## Foundations: what underwriting actually is

Start with everyday money. Suppose your neighbour is selling their house and needs the cash by Friday to close on a new one. A normal sale might take weeks and the final price is uncertain. So a property investor offers a deal: "I will pay you \$480,000 in cash on Friday, guaranteed. Then I'll resell the house at my own pace, and whatever I get above \$480,000 is my profit — and whatever I get *below* it is my loss." The neighbour has converted an uncertain future sale into certain cash today. The investor has taken on the risk of the resale in exchange for a margin. That is underwriting, exactly. Replace "house" with "ten million new shares" or "a \$1 billion bond," replace "property investor" with "investment bank," and you have the securities business.

The word "underwrite" comes from marine insurance at Lloyd's of London in the 1600s: a merchant who agreed to cover part of a ship's risk signed their name *under* the description of the voyage on a slip of paper. To underwrite is to put your name — and your money — under someone else's risk. A securities underwriter does the same thing for a fundraising: it signs its name under the deal and agrees to absorb some or all of the risk that the deal won't sell.

To make this concrete we need a few definitions, built from zero.

A **security** is a tradable financial claim — a share of stock (a slice of ownership) or a bond (a promise to repay borrowed money with interest). The party selling new securities to raise money is the **issuer**: a company doing an IPO, a government selling bonds, a firm raising fresh equity in a follow-on. The issuer's problem is that it is not in the business of finding thousands of buyers and negotiating with each one. It is in the business of making software, or building roads, or running an airline. It needs a specialist.

That specialist is the **underwriter** — almost always the capital-markets arm of an investment bank. (For the broader picture of what these banks do and how they make money, see [inside an investment bank](/blog/trading/finance/inside-an-investment-bank-how-they-make-money).) The underwriter does three jobs that are easy to confuse and worth separating cleanly:

1. **Origination** — winning the mandate, structuring the deal, and getting the paperwork (the prospectus) right.
2. **Distribution** — actually finding the buyers: calling institutional investors, gauging demand, building the order book, allocating shares.
3. **Risk-bearing** — agreeing to a contractual undertaking about *how much of the deal's failure the bank absorbs*. This is the part that distinguishes the three underwriting modes, and it is the part most people miss.

The third job is the crux. Two banks can do identical origination and identical distribution, but if one bought the deal outright and the other was merely trying its best, they are in two completely different businesses — one is a principal with capital at risk, the other is an agent collecting a commission. That distinction is so central that it has its own taxonomy.

## The three underwriting modes

There are three commitments an underwriter can make, and they sit on a spectrum from "the bank takes none of the risk" to "the bank takes *all* of it, overnight, before it has even told most buyers the deal exists."

![Matrix comparing firm commitment best efforts and bought deal by who bears risk price and use](/imgs/blogs/underwriting-and-the-syndicate-who-takes-the-risk-2.png)

### Firm commitment — the bank buys the whole deal

In a **firm-commitment** underwriting — by far the most common form for large IPOs and investment-grade bonds — the underwriter *buys the entire offering from the issuer at a fixed price* and then resells it to investors. The issuer is made whole at signing: it gets a guaranteed pool of cash regardless of what happens next. The bank now owns the inventory and bears the full risk that it can't resell at a profit.

The mechanics are worth slowing down on, because the timing is what makes the risk real. On pricing night, the issuer and the lead underwriter agree a price — say \$20 a share for 25 million shares, a \$500 million deal. The bank signs an underwriting agreement to buy those shares at \$20 minus the gross spread (its fee). It then sells to investors at \$20. If everything is sold the next morning, the bank simply collects its spread. But if the market sours overnight — a bad jobs number, a competitor's profit warning, a geopolitical shock — and the stock opens at \$18, the bank still owes the issuer the agreed proceeds. It is now sitting on shares worth \$18 that it is contractually committed to having paid \$20 for. That gap is the bank's loss.

#### Worked example: a firm commitment that breaks

A bank firm-commits a \$500 million IPO: 25 million shares at \$20. The gross spread is 7%, so the bank's all-in fee is `0.07 × \$500,000,000 = \$35,000,000`, and it pays the issuer `\$500,000,000 − \$35,000,000 = \$465,000,000`. The bank's effective cost per share is `\$465,000,000 / 25,000,000 = \$18.60`.

Now the deal "breaks" — bad news hits and the stock trades to \$18 before the syndicate can place all the stock. Suppose the bank has placed 15 million shares at \$20 (collecting `15,000,000 × \$20 = \$300,000,000`) but is stuck with the remaining 10 million shares, which it must dump into a falling market at \$18, raising `10,000,000 × \$18 = \$180,000,000`. Total received: `\$300,000,000 + \$180,000,000 = \$480,000,000`. Against a cost of \$465 million, the bank still nets `\$480,000,000 − \$465,000,000 = \$15,000,000` — less than the \$35 million spread it expected, but positive, because the \$18.60 cost basis gave it a cushion.

Push it harder: if the *whole* deal collapses to \$18 before the bank places anything, it sells all 25 million shares at \$18 = `\$450,000,000` against a \$465 million cost — a loss of `\$465,000,000 − \$450,000,000 = \$15,000,000`. The spread it hoped to earn (\$35M) has turned into a \$15M loss, a \$50 million swing. **The lesson: in a firm commitment, the spread is not a guaranteed fee — it is a cushion against exactly the inventory loss the bank just absorbed.**

This is why firm commitment is the *premium* product the issuer pays for. The issuer is buying certainty. The bank is selling insurance on the placement, and the spread is the premium. When you read that an IPO was "fully underwritten," that is the phrase — it means a bank put its balance sheet behind the proceeds.

### Best efforts — the bank is only an agent

At the opposite end is **best efforts**, also called an *agency* underwriting. Here the bank makes no promise about proceeds at all. It agrees only to *try* — to use its best efforts to sell as much of the deal as it can, at the agreed price, taking a commission on whatever it places. If half the deal goes unsold, that is the issuer's problem, not the bank's. The bank never owns the securities; it is a broker, not a principal.

Best efforts is used when nobody is willing to guarantee the deal: small, speculative, or early-stage offerings; companies the market may simply reject; deals where the issuer would rather risk an undersold offering than pay the steep premium for a firm commitment. A close cousin is the **all-or-none** offering, where the deal only closes if 100% is sold — otherwise every investor gets their money back and the deal is cancelled. Both protect the *bank's* capital entirely; the risk stays with the issuer.

The everyday-money analogy: best efforts is the estate agent who lists your house for a 3% commission and will work hard to sell it, but who never offers to *buy* it from you. Firm commitment is the property investor who hands you cash on Friday. You pay a lot more for the second deal, because the second person is taking a risk the first never touches.

#### Worked example: best efforts vs firm commitment for the issuer

A small biotech wants to raise \$50 million by selling 5 million shares at \$10. Compare the two modes.

Under **best efforts** at a 6% commission, the bank sells whatever it can. Say demand is soft and it places only 3.5 million shares. The issuer raises `3,500,000 × \$10 = \$35,000,000`, pays a commission of `0.06 × \$35,000,000 = \$2,100,000`, and nets `\$32,900,000` — and walks away \$15 million short of its target, with no recourse.

Under a (hypothetical) **firm commitment** at a 9% spread, the bank would guarantee the full \$50 million: the issuer nets `\$50,000,000 × (1 − 0.09) = \$45,500,000` no matter what, and the bank eats the unsold 1.5 million shares. But for a speculative biotech, no bank may *offer* a firm commitment at any reasonable price — the inventory risk is too high. **The lesson: the choice of underwriting mode is really a negotiation over who is willing to hold the inventory risk, and at what price; weak issuers often can't buy certainty at all.**

### The bought deal — a firm commitment, overnight

The **bought deal** (or block trade) is firm commitment turned up to its most aggressive setting. Instead of a multi-week marketing process, a bank agrees to buy an entire block of securities — typically a follow-on equity offering or a large secondary sale by an existing shareholder — *in a single overnight transaction*, at a price agreed before the market has been canvassed at all. The bank prices it after the close, takes the whole block onto its book, and then spends the next morning placing it with institutions, usually at a small discount to the last close.

Bought deals are competitive: when a company or a selling shareholder wants to move a large block fast and discreetly, several banks bid for the right to buy it, and the issuer takes the best price. The winning bank has, in a matter of hours, committed hundreds of millions of dollars of its own capital with no marketing cushion. If the stock gaps down the next morning, the bank is long a falling block it overpaid for.

#### Worked example: the overnight risk of a bought deal

A large shareholder wants to sell a \$300 million block of a liquid large-cap, last traded at \$50 (6 million shares). After the close, Bank A wins a competitive bought deal by bidding \$48.75 — a 2.5% discount to the last price. Bank A now owns 6 million shares at a cost of `6,000,000 × \$48.75 = \$292,500,000` and plans to place them at, say, \$49.25 the next morning.

If the placement goes well at \$49.25, Bank A grosses `6,000,000 × \$49.25 = \$295,500,000`, a profit of `\$295,500,000 − \$292,500,000 = \$3,000,000` for one night's risk — a thin 1.0% margin on capital deployed. But suppose overnight the broad market drops 4% and the stock opens at \$47. Now the block is worth `6,000,000 × \$47 = \$282,000,000`, and Bank A is staring at a paper loss of `\$292,500,000 − \$282,000,000 = \$10,500,000` — three and a half times the profit it was playing for. **The lesson: a bought deal compresses all the placement risk of a firm commitment into a single overnight window with no marketing buffer — the bank is paid a thin margin to be exposed, briefly, to the entire block.**

Bought deals exist because speed and certainty have value to sellers, and because a bank with confidence in its distribution can usually place a liquid block before the risk has time to bite. But they are precisely the trades where an investment bank can lose tens of millions in a single night, and they are why a bank's equity-capital-markets desk is, on bought-deal nights, indistinguishable from a proprietary trading desk.

## Why a deal is syndicated

Here is a puzzle. If firm-commitment underwriting is so risky — if a single bad overnight can cost a bank tens of millions — why would any bank ever firm-commit a \$10 billion mega-IPO? The answer is that no single bank does. It builds a **syndicate**: a temporary alliance of banks that jointly underwrite the deal, each taking a slice of the risk and a slice of the fee.

The everyday-money analogy is a group of friends buying a house to flip. None of them can afford the whole \$2 million on their own, or wants to bear the whole loss if it doesn't sell. So they split it: one puts up \$1 million and runs the renovation, two more put up \$400,000 each, and a couple of others kick in \$100,000 apiece and help find buyers. They share the upside in proportion, and — crucially — they share the *downside* the same way. That is a syndicate.

There are four reasons deals are syndicated, and all four matter:

1. **Risk-sharing.** A \$10 billion firm commitment split across a syndicate of fifteen banks is fifteen manageable exposures rather than one career-ending one. No single balance sheet has to carry the whole deal.
2. **Distribution reach.** Each bank brings its own roster of investor clients. One bank might be strong with US mutual funds, another with European insurers, a third with Asian sovereign wealth funds, a fourth with retail brokerage. Together they can canvass a far wider pool of demand than any one of them alone — and demand breadth is what gets a big deal sold without breaking the price.
3. **Research coverage.** When a bank joins a syndicate, its equity research analysts typically *initiate coverage* of the newly public company. More banks in the syndicate means more analysts publishing on the stock after the IPO, which means more visibility and (the issuer hopes) more durable secondary-market support.
4. **League-table credit.** Banks are ranked in published **league tables** by how much underwriting business they do. A spot on a marquee deal — even a junior one — earns league-table credit that the bank uses to win the *next* mandate. Reputation compounds, so banks will join deals partly for the standing it confers.

The structure of the syndicate is a hierarchy, and the titles tell you exactly who does what and who bears what.

At the top is the **lead bookrunner** (often just "the bookrunner" or "lead-left," from its position on the deal's cover). It runs the show: it owns the relationship with the issuer, builds and controls the order book, drives the pricing, and decides the allocation. On large deals there may be several **joint bookrunners** sharing the top-line work — and the issuer's secret hope is that having three or four big banks all incentivised to make the deal succeed produces better distribution than one.

Below the bookrunners sit the **co-managers**: banks that take a smaller economic slice and a smaller role, often included for their research coverage, their distribution into a particular investor segment, or a banking relationship the issuer wants to reward. Below them is the **selling group**: brokers who help place shares but take *no underwriting risk at all* — they are pure distribution, paid only the selling concession on what they sell. The selling group is to the syndicate what the best-efforts agent is to a firm commitment: a salesforce, not a risk-bearer.

#### Worked example: how a syndicate splits the risk

A \$2 billion IPO is firm-committed by a syndicate. The lead bookrunner takes 40% of the deal (\$800 million of underwriting liability), two joint bookrunners take 20% each (\$400 million each), and three co-managers split the remaining 20% (about \$133 million each). If the deal breaks and the syndicate eats a \$100 million inventory loss, the loss is shared in those proportions: the lead absorbs `0.40 × \$100,000,000 = \$40,000,000`, each joint bookrunner `0.20 × \$100,000,000 = \$20,000,000`, and each co-manager about `0.0667 × \$100,000,000 ≈ \$6,700,000`. **The lesson: the title on the cover is also a risk allocation — the bank with the biggest name on the deal is also the one with the most capital on the line if it goes wrong.**

### How the syndicate actually runs the deal

The syndicate is not just a list of names on the cover of the prospectus; it is a working machine with a clear division of labour during the live deal. Two documents glue it together. The first is the **agreement among underwriters** (the "AAU"), the private contract between the syndicate members that sets each bank's percentage, names the lead as the syndicate's agent, authorises the stabilising activity, and spells out how fees and any losses are shared. The second is the **underwriting agreement** between the syndicate and the issuer, which is the one that actually commits the banks to buy the deal — and it is signed only on pricing night, after the book is built, because before then nobody knows whether the deal will sell at all.

During the marketing period, the lead bookrunner runs the **book** — the running tally of investor demand. As the roadshow proceeds, institutional investors phone in **indications of interest** ("we'd take a million shares up to \$22"), and the bookrunner records the size, the price limit, and the *quality* of each order. Quality matters enormously: a long-only mutual fund that intends to hold the stock for years is a far better order than a hedge fund that the desk suspects will flip its allocation for the first-day pop. The bookrunner uses that information to set the final price and — critically — to decide **allocation**: who gets how many shares. Allocation is discretionary, not pro-rata, and it is one of the lead's most powerful (and most scrutinised) levers. The bank tends to favour investors it believes will be stable holders, investors who gave useful price feedback, and investors it does repeat business with.

This is where the syndicate's *distribution reach* becomes concrete. A deal that needs to place \$2 billion of stock cannot lean on one bank's client list; it needs the lead's relationships *plus* the joint bookrunners' *plus* every co-manager's. Each member works its own investors, feeds the orders into the lead's book, and the lead assembles the whole picture. The co-managers and selling group are, in effect, extra salesforces pointed at corners of the market the lead can't reach alone — which is exactly why their reward is the selling concession, paid per share placed.

#### Worked example: why distribution reach is worth a syndicate

Suppose a lead bank, working alone, can reliably generate \$1.2 billion of high-quality demand for a deal from its own client roster. The issuer wants to raise \$2 billion. Selling the extra \$800 million by pushing harder into the lead's existing clients would mean taking weaker, price-sensitive orders — the kind that force the price down or flip on day one. Instead the lead adds two joint bookrunners that bring \$400 million of fresh demand each and three co-managers that bring \$100 million apiece: `\$1,200,000,000 + 2 × \$400,000,000 + 3 × \$100,000,000 = \$2,300,000,000` of demand against a \$2 billion deal — a 1.15× covered book of *quality* orders, with room to allocate selectively. **The lesson: syndication is not mainly about splitting one bank's demand into pieces; it is about *summing* several banks' distinct investor bases into a book deep enough to price the deal well.**

## The gross spread and how it splits

The underwriters get paid out of the **gross spread** (also called the *underwriting discount*): the difference between the price the public pays for the security and the price the issuer receives. If a stock is offered at \$20 and the issuer nets \$18.60, the gross spread is \$1.40 a share, or 7%. The spread is the underwriters' entire compensation, and it is the issuer's all-in cost of selling the deal.

Spread sizes are not random. US IPOs famously cluster around **7%** — a remarkably sticky number that academics (notably Hsuan-Chi Chen and Jay Ritter) have called the "seven percent solution," because so many mid-sized US IPOs price at exactly 7% that it looks less like competition and more like a focal point. Larger IPOs negotiate lower (a multi-billion-dollar deal might pay 2–4%); bond deals pay far less (an investment-grade corporate bond might cost 0.3–0.875%, a government bond a few basis points) because the placement risk is lower and the buyers are concentrated, repeat institutions. The riskier the placement and the smaller the deal, the fatter the spread.

The gross spread does not all go to one bank. It splits into three classic pieces, in a proportion that is roughly **20 / 20 / 60**:

![Stack showing the gross spread split into management underwriting and selling fees](/imgs/blogs/underwriting-and-the-syndicate-who-takes-the-risk-4.png)

- The **management fee** (~20%) goes to the bookrunner(s) for originating and managing the deal — winning it, structuring it, running the process. It is payment for the *origination* job.
- The **underwriting fee** (~20%) compensates the syndicate members for *bearing the risk* — for putting their balance sheets behind the firm commitment. It is split in proportion to each bank's underwriting liability, and it also funds the syndicate's stabilisation activities (more on that below). It is payment for the *risk-bearing* job.
- The **selling concession** (~60%) — the largest slice — goes to whoever *actually sells the shares*, in proportion to the shares they place. It is payment for the *distribution* job. A co-manager or a selling-group broker who places a big chunk of the deal earns selling concession even though it bore little risk and did none of the origination.

This split encodes a deep truth about the business: **distribution is where most of the money is.** The bank that wins the mandate and bears the risk gets a 40% cut (management + underwriting), but the 60% selling concession rewards the boots-on-the-ground work of finding buyers. A bank that is great at sales can earn well in a syndicate without ever being lead-left.

#### Worked example: splitting a \$35 million spread 20 / 20 / 60

Take the \$500 million IPO with a 7% gross spread = \$35 million. Split it:

- **Management fee:** `0.20 × \$35,000,000 = \$7,000,000`, to the bookrunner(s) for running the deal.
- **Underwriting fee:** `0.20 × \$35,000,000 = \$7,000,000`, split across the syndicate by underwriting liability — the lead taking the largest share, co-managers the least.
- **Selling concession:** `0.60 × \$35,000,000 = \$21,000,000`, paid out per share placed.

Now suppose a co-manager bore only 5% of the underwriting liability but, because it has a strong retail brokerage network, *placed* 15% of the shares. It earns about `0.05 × \$7,000,000 = \$350,000` of the underwriting fee but `0.15 × \$21,000,000 = \$3,150,000` of the selling concession — `\$3,500,000` in total, almost ten times its risk-based share. **The lesson: the fee structure pays for *selling*, not just for risk — which is why a bank with great distribution but a small balance sheet still has a seat at the table.**

So far we have the static picture: who signs up, who bears the risk, who gets paid. But underwriting has a *dynamic* dimension that plays out in the first days of trading — and it is the part that makes a primary-market deal quietly different from anything in the secondary market. It is called the greenshoe.

## The greenshoe and price stabilisation

Here is the problem the greenshoe solves. The lead underwriter wants the new security to trade *well* in its first days — ideally a little above the offer price, never below it. A deal that "breaks issue" (trades below its offer price on day one) is a black eye: it tells the market the bank mispriced, it angers the investors who bought at the offer, and it makes the next deal harder to sell. But the underwriter can't control the open market. What it *can* do is arrange, in advance, to have ammunition to lean against early weakness — and the tool for that is the **over-allotment option**, universally nicknamed the **greenshoe** after the Green Shoe Manufacturing Company, whose 1963 IPO first used it.

The mechanism sounds like sleight of hand the first time you meet it, so let's build it step by step.

![Graph of the greenshoe mechanism from selling 115 percent to closing the short](/imgs/blogs/underwriting-and-the-syndicate-who-takes-the-risk-6.png)

The greenshoe is an option, granted by the issuer to the underwriters, to buy **up to 15% more shares** at the offer price, exercisable for about 30 days after the IPO. Armed with that option, the syndicate does something that looks impossible: it *sells 115% of the deal*. On a 100-share deal, it allocates 115 shares to investors. Where do the extra 15 shares come from? The syndicate doesn't own them — it has gone **short** 15% of the deal. It owes 15 shares it doesn't have. The greenshoe is its safety net for covering that short, and which way it covers depends on what the price does:

- **If the stock falls below the offer price**, the syndicate covers its short by *buying shares in the open market*. Those purchases are real demand hitting the order book — and they show up right when the stock is weak, which **cushions the fall and supports the price near the offer**. The syndicate bought back its 15% short cheaply (below offer), so it makes money on the short *and* it has propped the price. It does **not** exercise the greenshoe, because it covered in the market instead.
- **If the stock holds or rises above the offer price**, buying back in the market would mean covering the short at a loss (paying more than the offer to close a short it opened at the offer). So instead the syndicate **exercises the greenshoe**: it buys the extra 15% from the issuer at the offer price, delivering those shares to the investors it oversold to. The short is covered at zero P&L on the share price, the issuer raises 15% more money, and everyone is happy.

The elegance is that the *same structure* — selling 115% and going short 15% — produces a stabilising buyer when the stock is weak and a clean way to upsize when it is strong. The syndicate is never forced to buy a rising stock at a loss, and it always has a tool to defend a falling one. The bank running this is the **stabilising agent** (usually the lead bookrunner), and its activity is legal, disclosed, and regulated — in the US under Regulation M, which permits stabilising bids strictly to support a new issue's price, within tight rules, precisely because the alternative (an unsupported deal whipsawing on day one) is worse for everyone.

#### Worked example: greenshoe stabilisation P&L

A \$500 million IPO prices 25 million shares at \$20. The syndicate over-allots 15%: it sells `25,000,000 × 1.15 = 28,750,000` shares to investors, collecting `28,750,000 × \$20 = \$575,000,000` from buyers, but it only has 25 million real shares. It is short `3,750,000` shares.

Now the stock breaks to \$18. The stabilising agent buys back the 3.75 million short shares in the open market at \$18: it pays `3,750,000 × \$18 = \$67,500,000`. It had effectively "sold" those shares at \$20 (collecting `3,750,000 × \$20 = \$75,000,000`). The trading gain on covering the short is `\$75,000,000 − \$67,500,000 = \$7,500,000` — and that \$7.5 million of buying pressure hit the tape exactly when the stock was weak, propping it up. The syndicate does *not* exercise the greenshoe (the issuer still sold only its 25 million shares).

Contrast the rising case: the stock trades to \$22. Covering in the market would cost `3,750,000 × \$22 = \$82,500,000` to close a short opened at \$75,000,000 — a \$7.5 million loss. So the agent exercises the greenshoe instead: it buys 3.75 million shares from the *issuer* at the \$20 offer (`\$75,000,000`), covers the short at zero share-price P&L, and the issuer pockets an extra \$75 million of proceeds. **The lesson: the greenshoe is structured so the syndicate is always indifferent or better off — it earns a trading gain (and stabilises) when the stock falls, and upsizes the deal cleanly when it rises.** (Note that any stabilisation trading gain is generally returned to the syndicate's economics, not kept as a windfall; the point of the structure is price support, not speculation.)

This is exactly what Morgan Stanley's desk was doing in those first Facebook sessions — buying stock to cover an over-allotment short and lean against a sagging price. It is the most direct way a primary-market deal reaches into the secondary market and, for a few days, becomes one of its biggest participants.

## How issuance volume drives the whole business

All of this machinery — the syndicates, the spreads, the greenshoes — only fires when there are deals to do. And deal volume is wildly cyclical, which is why investment-bank underwriting revenue swings violently from year to year. The clearest way to see it is the dollar volume of US IPOs.

![US IPO proceeds by year from 2014 to 2024 with 2021 and 2022 highlighted](/imgs/blogs/underwriting-and-the-syndicate-who-takes-the-risk-3.png)

Look at the cliff. US IPO proceeds went from roughly \$78 billion in 2020 to about **\$142 billion in 2021** — a euphoric, free-money, everything-goes-public year — and then *collapsed to about \$8 billion in 2022* as the Fed hiked rates and risk appetite evaporated. That is not a dip; it is a ~94% drop in one year. For an underwriting desk, 2021 was a feast and 2022 was a famine — and the same desk, the same bankers, the same syndicate relationships, produced both. The spread percentage barely moved; what moved was the volume of deals to charge it on.

The count of deals tells the same story even more starkly, because it strips out deal size.

![Number of US IPOs by year from 2014 to 2024](/imgs/blogs/underwriting-and-the-syndicate-who-takes-the-risk-7.png)

The US went from **397 traditional IPOs in 2021** to just **71 in 2022** — barely one new listing a week, down from more than seven. When the window is open, everyone rushes through it; when it slams shut, almost nobody can get a deal done at any price. This is the deepest fact about the primary market: **issuance is a confidence business, and confidence is regime-dependent.** A bank can have the best syndicate desk in the world and still earn almost nothing in a closed window. (For why the IPO *window* opens and shuts the way it does, and how price discovery works inside it, see [bookbuilding and price discovery](/blog/trading/capital-markets/bookbuilding-and-price-discovery-how-the-ipo-price-is-set).)

Equity gets the headlines, but it is a rounding error next to debt. Far more capital is raised by *borrowing* than by *selling ownership*, and the underwriting machine runs hardest in the bond market.

![US debt issuance by type in 2023 with Treasury dominating](/imgs/blogs/underwriting-and-the-syndicate-who-takes-the-risk-5.png)

US Treasury issuance — the government selling its debt — dwarfs every other category, running into the tens of trillions of dollars a year once you include the constant rollover of short-term bills. Corporate bonds and mortgage-backed securities each run well over a trillion dollars annually; municipal bonds and asset-backed securities are smaller but still enormous. The point for our purposes: **most underwriting is bond underwriting, not equity underwriting**, and most bond deals are *syndicated* (sold by a group of banks to institutional buyers) rather than firm-committed at IPO-style risk — except for government bonds, which are mostly sold by **auction** rather than underwritten at all.

The contrast between the two is instructive, because it shows that underwriting is a *choice*, not a law of nature. A government with a deep, captive base of buyers — pension funds and banks that *must* hold its bonds — doesn't need a bank to guarantee the proceeds; it can just hold an **auction** and let the buyers compete on price. The US Treasury sells trillions a year this way, taking competitive bids from primary dealers and letting the clearing yield fall out of the demand. There is no firm-commitment risk and no spread to pay, because there is no placement risk to insure against: the buyers are reliably there. A corporate issuer, by contrast, has no such captive base. Its bonds must be *sold*, investor by investor, which is why corporate debt is **syndicated**: a group of banks builds a book of orders (just like an equity deal) and prices the bond at a small **new-issue concession** — a slightly higher yield than the company's existing bonds — to entice buyers. The underwriting risk on an investment-grade bond is real but modest, which is why the spread is a fraction of an IPO's: the deal is easier to place and the buyers are concentrated, repeat institutions. (For how a bond deal is actually run — auctions, competitive vs non-competitive bids, syndication, the new-issue concession, primary dealers — see [how a bond is issued](/blog/trading/capital-markets/how-a-bond-is-issued-auctions-syndication-and-the-deal); for why a firm raises debt vs equity in the first place, see [debt vs equity](/blog/trading/capital-markets/debt-vs-equity-the-two-ways-to-raise-capital).)

The deeper point is that **the underwriting mode and the spread both scale with one thing: how uncertain it is that the buyers will show up.** A Treasury auction has near-zero placement uncertainty and pays no spread. An investment-grade bond has low uncertainty and pays a thin spread. A mid-cap IPO has high uncertainty and pays a fat 7%. A speculative micro-cap can't buy a firm commitment at any price and is forced into best efforts. Lay the whole primary market on a single axis — *certainty of demand* — and the entire taxonomy of underwriting falls into place along it.

The thread connecting all of this back to the series spine is worth stating plainly. Underwriting is the act of *creating* a security and getting it into the market — it is the primary market's core function. But an underwriter will only firm-commit a deal because it believes it can *resell* the inventory, and it can only resell because there is a deep, liquid **secondary market** standing ready to absorb shares. The greenshoe is the cleanest proof: the entire stabilisation mechanism is the underwriter operating in the secondary market to make the primary deal succeed. **Secondary-market liquidity is what makes primary issuance possible** — no bank would buy a \$500 million block if it couldn't sell it tomorrow. The primary and secondary engines are not separate; underwriting is the gear that meshes them.

## The deliberate underpricing problem

There is one more piece of the risk story that connects the spread, the syndicate, and the issuer's interests in an uncomfortable way: the **IPO pop**, the tendency for newly issued shares to jump on their first day of trading.

![Median first-day IPO pop in the US by year](/imgs/blogs/underwriting-and-the-syndicate-who-takes-the-risk-8.png)

A first-day pop sounds like good news, and for the investors who got an allocation at the offer price, it is — they make an instant paper gain. But think about who *paid* for that gain. If a stock is offered at \$20 and opens at \$27, the issuer sold its shares \$7 too cheap. On 10 million shares that is \$70 million of value that went to the lucky allocated investors instead of into the company's treasury — far more than the entire underwriting spread. That gap is called **"money left on the table,"** and in hot years (2020's median pop was over 40%) it dwarfs the fees.

Why does it happen? Partly because pricing a never-before-traded security is genuinely hard. But partly because the underwriter's incentives are not perfectly aligned with the issuer's. The bank wants the deal to trade up (a pop makes its institutional clients happy and makes the *next* deal easy to sell), it wants to avoid the catastrophe of breaking issue, and it allocates the underpriced shares to the very clients it wants to keep happy. A modest, deliberate underpricing is the bank's insurance against a failed deal — paid for with the issuer's money. The issuer tolerates it because the alternative (pricing aggressively and risking a break) is worse, and because the bank's relationships are what get the deal done at all.

This is the underwriting business in one tension: the issuer hires the bank to get the *highest* price, but the bank's whole risk-management instinct — and its loyalty to the buy-side clients it sells to again and again — pulls toward a *safe* price, one that pops. The spread is the visible fee; the pop is the much larger, invisible one. (For the mechanics of *how* that price gets set inside the book, and why auctions like Google's 2004 IPO are rare, see [bookbuilding and price discovery](/blog/trading/capital-markets/bookbuilding-and-price-discovery-how-the-ipo-price-is-set).)

## League tables and reputational capital

Why would a bank join a deal as a junior co-manager, doing real work for a sliver of the fee? Part of the answer is the selling concession it earns on what it places. But a larger part is something that never appears on the deal's economics: **league-table credit and reputation.**

Investment banks are ranked publicly — by Dealogic, Bloomberg, Refinitiv, and others — in **league tables** that tabulate, for each quarter and year, how much underwriting business each bank did: total proceeds, number of deals, market share, broken down by product (equity, debt, high-yield) and region. These tables are scoreboard, marketing brochure, and recruiting tool all at once. A bank that ranks top-three in global equity underwriting can walk into the *next* issuer's boardroom and say "we are the number-one bank for deals like yours" — and that claim, backed by the table, helps win the mandate. The league table is how past deals turn into future deals.

This creates a self-reinforcing loop that is central to how the business actually works. Reputation wins mandates; mandates produce more deals; more deals improve the league-table rank; a better rank wins more mandates. It is why the bulge-bracket banks stay at the top for decades, and why a bank will sometimes take a deal at a thin fee, or join a syndicate in a junior role, purely to keep its name on the scoreboard. **The deal economics are the short game; the league table is the long game.**

Reputation is load-bearing in a more subtle way too. When a bank firm-commits a deal and stands behind the proceeds, the issuer is trusting the bank's *judgement* about what the market will bear. When investors buy at the offer price, they are partly trusting that the bank wouldn't put its name on a deal it didn't believe in — because a bank that repeatedly brings deals that break issue or blow up afterward will find its calls going unanswered. The underwriter's reputation is, in effect, a quality stamp it lends to the issuer. That is why a bank guards its franchise carefully: it can survive losing money on a single bad deal, but it cannot survive being known as the bank whose deals you shouldn't buy. Reputational capital, accumulated deal by deal and tracked on the league tables, is the asset the whole underwriting business is built to protect.

#### Worked example: why a bank takes a thin-fee deal for the rank

A bank is offered a junior co-manager spot on a \$3 billion landmark IPO, with a 1% economic share — about `0.01 × (0.04 × \$3,000,000,000) = \$1,200,000` of a \$120 million spread. That is a modest fee for a top-tier bank. But the deal adds \$3 billion of *credited* equity-underwriting volume to its league-table total (co-managers typically receive full or substantial deal credit), which might lift it a rank or two in the quarterly standings — and a higher rank is the pitch that helps win a \$10 billion mandate next year, worth perhaps `0.40 × (0.025 × \$10,000,000,000) = \$100,000,000` in spread if the bank leads it. **The lesson: a bank will rationally take a \$1.2 million fee for the league-table credit that helps it chase a \$100 million one — reputation is an investment, not a vanity.**

## Common misconceptions

**"Underwriting just means the bank helps sell the shares."** Only in a best-efforts deal. In a firm commitment — the standard for large deals — the bank *buys the entire offering* and resells it, putting its own capital at risk. The whole point of the word is risk transfer, not sales assistance. A bank that only helps sell, with no risk, is running a best-efforts agency deal, and it is a fundamentally different and cheaper service.

**"The greenshoe is the bank making extra money on the side."** The greenshoe is a price-*stabilisation* tool, not a profit centre. When the syndicate buys back its over-allotment short below the offer, the resulting trading gain is generally returned to the deal's economics, and the buying happens precisely to support a weak stock. The bank's compensation is the gross spread; the greenshoe exists to make the deal *succeed*, which protects the bank's reputation and its next mandate. Treating it as a windfall misreads its purpose.

**"A bigger spread means a greedier bank."** Spreads scale with *risk and difficulty*, not greed. A 7% spread on a small, hard-to-place IPO and a 0.4% spread on a giant investment-grade bond can reflect identical competitive markets — the IPO is simply far riskier to underwrite and far harder to distribute. The sticky 7% on mid-cap US IPOs is a real puzzle, but the broad pattern (riskier, smaller deals cost more) is just the price of insurance.

**"The lead bank does all the work, so the others are just along for the ride."** The co-managers and selling-group brokers earn the bulk of the selling concession — 60% of the spread — precisely because *distribution* is most of the work and most of the value. A co-manager with a strong investor network can place more shares (and earn more) than a bookrunner with a weak one. The titles describe roles and risk shares, not a ranking of usefulness.

**"If a deal is fully underwritten, the issuer is guaranteed its money no matter what."** Mostly true — but underwriting agreements contain a **market-out clause** (and a force-majeure clause) that lets the underwriters walk away before closing if a major market dislocation occurs (a crash, a war, a systemic shock). The guarantee is firm against ordinary bad luck, not against a meltdown. The bank's commitment is real but not unconditional.

## How it shows up in real markets

**Facebook, May 2012.** As described at the top, the lead underwriter ran one of the most visible stabilisation campaigns in history, buying stock to defend the \$38 offer price as the deal sagged. It was a textbook (if ultimately unsuccessful) greenshoe-and-stabilisation operation: the syndicate had oversold the deal, was short, and used its buying to lean against the fall. It also became a cautionary tale about the limits of stabilisation — when the fundamental demand isn't there, a stabilising bid can slow a decline but not reverse it. Facebook eventually traded into the high \$20s before recovering over the following year. The episode is the clearest public demonstration that an underwriter's job does not end at the opening bell.

**Saudi Aramco, December 2019.** The largest IPO ever (about \$25.6 billion, later \$29.4 billion after the greenshoe) shows syndication at scale: a sprawling syndicate of global and local banks was assembled, but the deal was ultimately placed largely with domestic and regional investors, and a 15% greenshoe was exercised when the stock held above its offer. It is a reminder that even the biggest deals on earth run on the same plumbing — a lead, a syndicate, a spread, an over-allotment — just with more zeroes.

**The 2021-to-2022 IPO cliff.** As the proceeds and count charts show, the US IPO market went from a 397-deal, \$142-billion boom to a 71-deal, \$8-billion bust in a single year as rates rose. Underwriting desks that had staffed up for the boom found themselves with almost nothing to underwrite. This is the cyclicality of the primary market in its rawest form: the *capacity* to underwrite is permanent, but the *opportunity* to underwrite vanishes when confidence does. It is why investment-bank earnings are so volatile, and why a great franchise can post a terrible year through no fault of its own.

**The "seven percent solution."** Academic work found that the overwhelming majority of moderate-sized US IPOs (roughly \$20–80 million in the studied period) priced at *exactly* 7.0% gross spread — a clustering far too tight to be explained by independent competitive bargaining. The leading interpretations range from implicit coordination to a stable equilibrium where issuers accept a focal price in exchange for full service. Either way, it is a striking real-world case where a "competitive" fee turns out to be remarkably sticky — a useful caution against assuming markets always grind fees to the bone.

**The bought-deal block market.** On any given week, large shareholders — private-equity sponsors exiting a position, a founder diversifying, a government privatising a stake — sell big blocks via overnight bought deals. Banks compete fiercely to win them, sometimes bidding so tightly that the margin barely covers the overnight risk. Occasionally a block gaps down and the winning bank takes a real loss; more often the bank places it by morning and books a thin, fast profit. It is the purest expression of underwriting as risk-warehousing: capital committed for hours, paid a sliver, exposed to the whole move.

**The pulled deal — when underwriting becomes "no deal."** The risk in a firm commitment is not only that the bank eats a loss; it is that the deal might not happen at all. When the market turns hostile during the marketing window — the book isn't covered, or the price the issuer wants is far above what investors will pay — the lead and the issuer often choose to **pull the deal** rather than price it badly. WeWork's planned 2019 IPO is the famous case: as investors balked at the valuation and the governance, the deal was shelved days before it was due to price, and the company's funding crunch that followed nearly sank it. Pulling a deal protects the bank from underwriting an offering it can't place, but it is its own kind of damage: months of work unbilled, a public embarrassment for the issuer, and a dent in the syndicate's standing. It is the reminder that the underwriter's first job — long before stabilisation — is the judgement call of whether the deal can be sold at all, and that "we are not doing this deal today" is sometimes the most valuable advice a bank gives.

**Emerging-market and Vietnam deals.** In smaller and developing markets, the same architecture appears but the risk is sharper, because the secondary market the underwriter relies on to resell its inventory is thinner. In Vietnam, large equity offerings and state-enterprise privatisations are typically run by a lead securities firm with a syndicate of domestic brokers, and the deal's success often hinges on foreign-institutional demand that can be fickle — net foreign flows on the Ho Chi Minh exchange have swung from large net buying to heavy net selling year to year. A thin, flow-dependent secondary market makes firm-commitment underwriting riskier and rarer, and pushes more deals toward best-efforts or auction formats — a direct illustration of the series' core point that *secondary-market liquidity is what makes primary issuance possible.* Where the liquidity is shallow, the underwriter is less willing to put its balance sheet on the line, and the whole primary market downshifts accordingly.

## The takeaway: underwriting is the price of certainty

Strip away the jargon and underwriting is one idea: **somebody converts an issuer's uncertain future sale into certain cash today, and charges for bearing the risk in between.** Everything else — the three modes, the syndicate hierarchy, the 20/20/60 spread, the greenshoe — is machinery built to make that risk survivable and that certainty deliverable.

The single most useful lens is to always ask, of any deal, *who eats the loss if the buyers don't come?* In a firm commitment, the bank does, and it charges a fat spread as the premium for that insurance. In best efforts, the issuer does, and the bank is just a commissioned agent. In a bought deal, the bank does — overnight, with no marketing cushion, for the thinnest of margins. The syndicate exists so that no single bank has to answer "all of it." The greenshoe exists so the bank has a tool to defend the price in the days when that risk is most acute. The spread, split three ways, pays separately for winning the deal, for bearing the risk, and — most of all — for the unglamorous work of finding buyers.

And it all connects to the larger machine. The reason a bank will *dare* to firm-commit \$500 million of stock is that it trusts the secondary market to take that stock off its hands. The primary market that creates securities and the secondary market that trades them are joined at exactly this seam: underwriting is the act of crossing from one to the other, and the underwriter's willingness to cross is borrowed entirely from the liquidity waiting on the far side. Next time you see a company "go public," picture the syndicate behind it — the lead-left bank that signed its name under the risk, the co-managers fanning out to find buyers, and the stabilising desk quietly ready to step into the market on day one. That is the primary market actually working: turning a pile of new paper into cash for the issuer, by standing in the gap and taking the risk.

## Further reading & cross-links

- [The IPO process, end to end: from mandate to first trade](/blog/trading/capital-markets/the-ipo-process-end-to-end-from-mandate-to-first-trade) — the full storyline this post zooms into.
- [Bookbuilding and price discovery: how the IPO price is set](/blog/trading/capital-markets/bookbuilding-and-price-discovery-how-the-ipo-price-is-set) — how demand becomes a price, and why the pop happens.
- [How a bond is issued: auctions, syndication, and the deal](/blog/trading/capital-markets/how-a-bond-is-issued-auctions-syndication-and-the-deal) — underwriting on the (much larger) debt side.
- [Debt vs equity: the two ways to raise capital](/blog/trading/capital-markets/debt-vs-equity-the-two-ways-to-raise-capital) — why an issuer chooses what to underwrite in the first place.
- [Inside an investment bank: how they make money](/blog/trading/finance/inside-an-investment-bank-how-they-make-money) — where the underwriting desk sits in the firm.
- [Stock exchanges and clearinghouses](/blog/trading/finance/stock-exchanges-and-clearinghouses) — the secondary-market plumbing the underwriter relies on to resell its inventory.
