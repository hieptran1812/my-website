---
title: "Sell-Side Research: The Analyst, the Rating, and the Wall"
date: "2026-06-21"
publishDate: "2026-06-21"
description: "What sell-side equity and credit research is actually for, why ratings skew toward buy, and how the Chinese wall, the 2003 settlement, and MiFID II reshaped the information layer of capital markets."
tags: ["capital-markets", "sell-side-research", "equity-research", "analysts", "chinese-wall", "mifid-ii", "conflicts-of-interest", "intermediaries"]
category: "trading"
subcategory: "Capital Markets"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — Sell-side research is the information layer of capital markets: analysts publish models, price targets, and buy/hold/sell ratings that help the buy-side price the claims the primary market issues. Banks run it not because reports turn a profit but because research greases sales, trading, and banking mandates — which is exactly why it carries a built-in conflict.
>
> - A bank's research desk is usually a cost center; it earns its keep by feeding trading commissions and helping win a single \$35M IPO fee, not by selling reports.
> - Ratings skew hard toward "buy" — roughly half buys, only a single-digit share sells — because access, banking ties, and the fallout of a real sell all push analysts away from negativity.
> - The Chinese wall, the 2003 Global Analyst Research Settlement, and MiFID II unbundling are three waves of trying to fix that conflict — and each had side effects, especially thinner coverage of small caps.
> - The one number to remember: after MiFID II forced research to be priced separately, European research budgets fell by roughly 20–30% and small-cap coverage shrank — the price of honesty was less information.

## A morning in 2000

On a December morning in 2000, a retail investor in Ohio opened his brokerage statement and saw that the internet stock he had bought a year earlier — on the strength of a glowing "Buy" rating from a famous Wall Street analyst — was down 80%. The analyst's price target, once \$200, was a memory. What he did not know, and would not learn until a state attorney general subpoenaed the analyst's email two years later, was that the same analyst had privately called the stock "a piece of junk" while publicly rating it a top pick.

That gap — between what the analyst published and what the analyst believed — is the whole story of sell-side research compressed into one email. Research is supposed to be the part of the capital-markets machine that tells the truth about a company so that savers can decide whether to fund it. But the people who produce that research work inside banks that make most of their money in other ways: by trading the stock, by selling it to clients, and above all by winning the right to underwrite the company's next stock or bond deal. The research is honest most of the time. The conflict is structural all of the time.

This post is about that information layer — what research is for, why it leans optimistic, how a wall inside the bank is supposed to keep the conflict in check, and what two decades of scandal and reform did to it.

![Diagram of the Chinese wall separating the public research side from the private banking side inside a bank](/imgs/blogs/sell-side-research-the-analyst-the-rating-and-the-wall-1.png)

## Foundations: what sell-side research actually is

Before any jargon, an everyday-money analogy. Suppose you are deciding whether to lend \$10,000 to a friend's restaurant. You would want someone who has studied the restaurant's books, talked to the owner, modeled how many covers it needs to break even, and given you a verdict: lend, wait, or run. Sell-side research is that someone, scaled up to thousands of public companies and bonds, and published to the whole market.

A few terms, defined from zero:

- **The sell side** is the set of investment banks and brokers that *create and sell* securities and services to investors. They are called "sell side" because they are selling — selling shares in an IPO, selling bonds, selling trade execution, and selling ideas. (The **buy side** — pension funds, mutual funds, hedge funds — *buys* those securities. We cover them in the sibling post on the buy-side.)
- **An analyst** is a person on the sell side who covers a defined set of companies (their "coverage universe" — often 10–20 names in one sector, say semiconductors or US banks) and publishes research on them.
- **A research note** is the analyst's written product: a financial model forecasting the company's revenue and earnings, a written thesis, a **price target** (where the analyst thinks the stock will trade in 12 months), and a **rating** — a one-word verdict, usually **buy**, **hold**, or **sell** (banks dress these up as "overweight / neutral / underweight" or "outperform / market-perform / underperform", but it is the same three buckets).

There are two flavors. **Equity research** covers stocks: it forecasts earnings and sets a price target and a buy/hold/sell rating. **Credit research** covers bonds: instead of "will the stock go up?" it asks "will this borrower pay me back, and is the yield enough for the risk?" — its verdict is an *over/underweight* on the bond and a view on the credit spread. We will mostly speak in equity terms, but the conflicts are the same on both sides.

The crucial thing to understand — and the thing that explains every conflict that follows — is **how the research desk gets paid.** It does not. Not directly. A research report is given away free to the bank's clients. So why does a bank employ hundreds of analysts and pay the stars millions?

#### Worked example: why a bank tolerates a money-losing research department

Suppose a bank runs an equity research department covering 400 companies with 40 analysts. Loaded cost — salaries, data terminals, travel, compliance — runs about \$1.5M per analyst, so the department costs roughly:

$$
40 \times \\$1{,}500{,}000 = \\$60{,}000{,}000 \text{ per year.}
$$

The department sells nothing. On its own P&L it loses \$60M a year. So how does it pay?

Two ways. First, **trading commissions.** When a buy-side fund values a bank's research, it routes trades through that bank's trading desk as a thank-you. If research helps the desk capture, say, an extra \$45M a year in commissions, the department is already net positive. Second — and this is the one that creates the conflict — **banking mandates.** When a company picks a bank to run its IPO, having a respected analyst who covers the sector and will support the stock afterward is part of the pitch. Win one mid-size IPO and the math changes entirely:

$$
\text{IPO size} = \\$500{,}000{,}000, \quad \text{underwriting fee} = 7\% = \\$35{,}000{,}000.
$$

A *single* \$35M IPO fee covers more than half the entire research department's annual cost. Win two or three such mandates a year — partly because your analyst is known and liked in that sector — and the "money-losing" research department is one of the best investments the bank makes. **Intuition: research is a loss leader that sells the bank's two real products — execution and underwriting — which is precisely why the analyst is never a neutral party.**

This is the spine of the whole series in miniature. A capital market turns savings into investment through a **primary market** (which creates securities to raise money) and a **secondary market** (which trades them). Research sits on top of both: it is the information layer that helps investors price the claims the primary market issues, and it is paid for by the trading volume of the secondary market and the fees of the primary market. To see where the research desk plugs into the rest of the [investment bank](/blog/trading/capital-markets/inside-an-investment-bank-ecm-dcm-ma-and-trading), keep that org chart in mind: research is public-facing; ECM, DCM, and M&A are private and deal-driven.

It is worth pausing on *why* a free product can be so valuable to its producer, because the answer is the engine of every conflict that follows. The buy-side cannot route its trades to a bank that has given it nothing; trading desks compete on more than just price and speed, and good research is the relationship glue. Before 2018 (and still in the US), the way a fund repaid a bank for research it valued was simply to send that bank trade orders. The commissions on those orders are the bank's compensation for the research — an indirect, untracked, "soft-dollar" arrangement. So a research department that produces work the buy-side respects pulls in order flow, and order flow is revenue. The chain is: good analyst → grateful fund → more trades through our desk → commissions that cover the analyst's cost many times over. None of that requires anyone to sell a single report.

The banking link is subtler and more dangerous. No one in a compliant bank ever says "rate this company a Buy so we win the mandate" — the wall and the 2003 settlement forbid it explicitly. But a company choosing an underwriter looks at, among other things, whether the bank's analyst is respected in the sector and whether that analyst will provide ongoing coverage after the deal. An analyst who is known, who attends the conferences, who has a constructive relationship with the management team, is an asset in the banking pitch. The analyst does not have to be told to be friendly; the incentive structure makes friendliness the rational default. That is what makes the conflict *structural* rather than a matter of a few bad actors: even a scrupulously honest analyst operates inside a machine that rewards optimism and punishes negativity.

![Pipeline showing research turning issuer disclosure into ratings the buy-side uses to fund primary issuance](/imgs/blogs/sell-side-research-the-analyst-the-rating-and-the-wall-5.png)

## What an analyst produces, concretely

An analyst's day is built around a few outputs. The **earnings model** is a spreadsheet that projects the company's income statement, balance sheet, and cash flow — often quarter by quarter for two years and annually beyond. The **estimate** is the single number the market cares about most: forecast earnings per share (EPS) for the next quarter and year. The Street's average of all analysts' estimates is the **consensus**, and a company "beating" or "missing" consensus on earnings day is a headline because the stock often moves several percent on the surprise.

A second product, often more important to the buy-side than the rating, is the **estimate revision**. Markets are forward-looking, so what moves a stock is not the level of an analyst's forecast but the *change* in it. When an influential analyst raises next year's EPS estimate from \$4.00 to \$4.50, that 12.5% upward revision often moves the stock more than a fresh Buy rating would, because it signals that the analyst's model of the business has genuinely improved. Sophisticated funds subscribe to revision data precisely to trade these changes — a discipline sometimes called "earnings-revision momentum." It is the cleanest example of why the *content* of research, not its verdict, is the part with real information.

A third product is **initiation of coverage.** When a bank's analyst picks up a new name — say after the company's IPO, or because the sector has grown important enough to staff — the first published note is the "initiation," typically a long, comprehensive piece laying out the full thesis, model, and target. Initiations matter because they expand the universe of companies the market is actively pricing. A company that no analyst has initiated coverage on is, in a real sense, less visible to capital — a point that becomes central when we get to MiFID II.

The **price target** is where the analyst thinks the stock should trade in 12 months. It is the most visible — and most misunderstood — number in research. A target is not a promise; it is the output of a valuation. (How that valuation is built — discounted cash flow, multiples, sum-of-the-parts — belongs to equity research as a discipline; we link out to it rather than re-derive it here.) What matters for our purpose is the *implied return* a target advertises.

#### Worked example: a price target's implied return

A stock trades at \$80. An analyst initiates coverage with a **Buy** and a 12-month price target of \$100. The implied price appreciation is:

$$
\frac{\\$100 - \\$80}{\\$80} = \frac{\\$20}{\\$80} = 25\%.
$$

Add a 2% dividend yield and the implied **total return** is about 27% over a year. That is the "edge" the analyst is advertising: follow this Buy and the model says you make ~27%.

Now flip it. For the rating to be a Hold rather than a Buy, many banks use a band — say a Buy needs implied upside above 10–15%, a Hold sits within ±10%, a Sell needs downside beyond 10%. So if the same analyst's honest target were \$84 (5% upside), the rules say it should be a **Hold**. The distance between a \$100 "Buy" target and an \$84 "Hold" target is the distance between a recommendation that wins the analyst attention, corporate access, and banking goodwill, and one that wins none of those. **Intuition: the price target is where the analyst's incentives and the analyst's spreadsheet meet — and the incentives all push the target up.**

## The ratings and their famous skew

If ratings were honest signals, roughly a third of stocks should be buys, a third holds, a third sells — the market cannot have everything be a bargain. The reality is wildly lopsided. Across the sell side, buys typically run around half of all ratings, holds about 40%, and **sells are usually in the single digits** — often 5–10%. Whole sectors go years with almost no sell ratings outstanding.

![Stylised distribution of sell-side ratings showing about half buys, most of the rest holds, and few sells](/imgs/blogs/sell-side-research-the-analyst-the-rating-and-the-wall-3.png)

Why the skew? Three forces, all structural:

1. **Access.** Analysts need to talk to company management — at conferences, on calls, in one-on-one meetings that they arrange for their buy-side clients. A company that gets a "Sell" can quietly freeze the analyst out: no calls returned, no meetings, no invitation to the analyst day. An analyst frozen out of a major name in their sector is suddenly less useful to clients than rivals who still have access. So the sell rating costs the analyst the very relationship that makes them valuable.

2. **Banking relationships.** A "Sell" on a company is a near-guarantee that the bank will never win that company's underwriting or M&A business. No CFO hires the bank whose analyst is publicly telling the world the stock is going to fall. Given that a single mandate can be worth tens of millions (recall the \$35M IPO fee above), the implicit pressure not to alienate a potential client is enormous — even when the wall (next section) forbids anyone from saying so out loud.

3. **The awkwardness of a sell.** A "Hold" has become the polite code for "sell." Everyone on the buy side knows that when a respected analyst moves a name from Buy to Hold, the real message is "get out." So the analyst can express negativity without the personal and professional fallout of the dreaded word. The result is a rating scale that has effectively lost its bottom: the action happens between Buy and Hold, and the true Sell is reserved for companies already in freefall.

There is a fourth, quieter force worth naming: **career risk.** An analyst who issues a lone Sell on a popular stock is making a public, dated, falsifiable bet against the crowd. If the stock keeps rising for another year — as overvalued stocks often do — the analyst looks wrong in public for months, clients complain, and the company's investor-relations team works the phones to discredit them. The asymmetry is brutal: a Buy that goes up earns mild credit, a Buy that goes down is forgiven as "the market," but a Sell that goes up is a visible, personal failure. Faced with that payoff structure, the rational analyst saves the Sell for cases where being wrong is nearly impossible — which means by the time the Sell appears, the bad news is usually already in the price.

Credit research has its own version of the skew. A credit analyst rarely publishes "this bond will default"; instead they nudge a recommendation from *overweight* to *market-weight* and widen their fair-value spread, the bond-market equivalent of a coded Hold. And because the issuer is often a current or prospective bond-underwriting client of the bank, the same banking-relationship pressure applies. The skew is universal across the sell side because the incentive that creates it — the producer is paid by the side that wants the security to look good — is universal.

#### Worked example: the cost of a coded "Hold"

A fund manager holds \$5,000,000 of a stock at \$50 (100,000 shares). An analyst downgrades it from Buy to **Hold** — not Sell — and trims the target from \$60 to \$48. A sharp PM reads "Hold" as "sell" and exits over a week at an average of \$49. Three months later the stock is \$38 after a bad quarter the analyst privately feared but would not put in writing as a Sell.

The PM who decoded the Hold avoided:

$$
(\\$49 - \\$38) \times 100{,}000 = \\$11 \times 100{,}000 = \\$1{,}100{,}000 \text{ of losses.}
$$

A less experienced investor who took "Hold" literally and kept the position lost that \$1.1M. **Intuition: the skew does not just inflate the buys — it hides real information inside euphemisms that only insiders decode, which transfers value from naive holders to sophisticated ones.**

## The Chinese wall: policing the conflict

Given that the same bank both rates a company and wants to underwrite its deals, regulators and banks built an **information barrier** — historically called the "Chinese wall," now more often the "information barrier" or "ethical wall" — between the two sides.

On the **public side** sit research, sales, and trading: people whose job is to talk to the market and who may only act on public information. On the **private side** sit the bankers — equity capital markets (ECM), debt capital markets (DCM), and M&A — who routinely possess **material non-public information** (MNPI): the secret that a client is about to announce a takeover, a bond deal, a profit warning. The wall's job is to make sure MNPI never crosses from the private side to the public side, because if an analyst or trader acted on it, that would be insider dealing — a crime.

Refer back to the cover figure: the wall is the amber column down the middle. Bankers (lavender, private side) hold MNPI; the barrier stops it crossing west. Analysts (blue, public side) build views only from public disclosure and publish a report (green) that the buy-side reads. The dotted arrow — the issuer telling its banker about a coming deal — is legitimate; the dashed arrow that the wall blocks with a hard bar is the one that would be a crime if it got through.

How is the wall actually policed? Not with bricks but with controls:

- **Physical and system separation** — different floors, separate document systems, access controls so research literally cannot open the deal files.
- **The watch list and the restricted list.** When a bank starts working on a confidential deal for a company, that company goes on a private **watch list** (compliance monitors trading and research for leaks). When the deal becomes public or sensitive enough, it moves to the **restricted list**: research must stop publishing on it and the firm restricts proprietary trading, precisely to avoid the appearance that public output is being driven by private knowledge.
- **Wall-crossing procedures.** Occasionally a banker legitimately needs to bring a public-side person "over the wall" — for example, to sound out an analyst on a sensitive matter. This is logged, time-stamped, and the crossed person is restricted from trading until the information is public. It is the controlled exception that proves the rule.

The wall is genuinely effective at stopping the *crime* (trading on secrets). What it was never designed to stop — and historically failed to stop — was the softer, legal conflict: an analyst writing optimistically to please a company the bank wanted as a client. That failure produced the defining scandal of the field.

It helps to be precise about *why* a barrier between two parts of one company is even necessary, because it reveals what kind of institution a bank is. A bank is not one business; it is several businesses that happen to share a brand, a balance sheet, and a building, and whose interests routinely conflict. The banking division's interest is to win and execute deals for issuers. The trading division's interest is to make markets and trade for its own and clients' accounts. Research's interest is — in theory — to tell investors the truth. Sales' interest is to keep institutional clients trading. Left unwalled, these would contaminate each other constantly: a banker would lean on an analyst, a trader would front-run a deal, a salesperson would whisper a coming downgrade to a favored client. The wall is the institutional admission that the bank's own divisions cannot be trusted to share information freely without breaking the law or the public's trust. It is, in effect, a company protecting the market from itself.

The cost of the wall is real friction. Information that would make the bank smarter is deliberately blocked from flowing. An analyst cannot use what the banking team knows, even though it would improve the research, because using it would be a crime. The bank accepts being dumber on the public side as the price of being allowed to operate both sides at all. That trade-off — internal inefficiency in exchange for market integrity — is the same logic that runs through all of disclosure-based regulation: the system is built to constrain the insider, not to make the insider's life easy.

## The conflicts and scandals: the dot-com research settlement

In the late 1990s, the conflict ran wild. Investment banking fees from the internet boom were enormous, and analysts became part of the sales pitch. The two emblematic figures:

- **Henry Blodget**, an internet analyst at Merrill Lynch, became famous for sky-high targets on dot-com stocks. New York Attorney General Eliot Spitzer's 2002 investigation surfaced internal emails in which Blodget and colleagues privately disparaged stocks — calling them junk, a "powder keg," a "piece of garbage" — that they were rating Buy or Accumulate in public. The published ratings, the emails suggested, were calibrated to win and keep banking business, not to inform investors.
- **Jack Grubman**, the star telecom analyst at Salomon Smith Barney (Citigroup), was even more entangled: he sat in on his banking clients' board-level discussions and famously kept a "Buy" on WorldCom almost to its collapse. He was later accused of upgrading AT&T partly to curry favor in an unrelated personal matter. He was barred from the securities industry for life and fined \$15M.

The regulatory response was the **2003 Global Analyst Research Settlement** (formally finalized in April 2003), struck between regulators (the SEC, NASD, NYSE, and state regulators led by New York) and ten of the largest banks. The headline numbers: about **\$1.4 billion** in total penalties, disgorgement, and funding, including roughly \$450M earmarked to buy *independent* third-party research for investors and \$80M for investor education.

The structural reforms mattered more than the fines:

- **Sever the pay link.** Analyst compensation could no longer be tied to specific investment-banking deals they helped win. An analyst's bonus could not be a cut of the IPO fee their Buy rating helped land.
- **No banker influence over research.** Bankers were barred from reviewing or approving research reports before publication, and from promising favorable research to win mandates.
- **Disclosure on every report.** Research had to disclose the bank's banking relationship with the covered company, its holdings, and the firm-wide distribution of its buy/hold/sell ratings — so a reader could see, for instance, "this firm rates 52% of its covered companies Buy and 4% Sell."
- **Physical separation and the firewall** were hardened and made explicit.

#### Worked example: the math behind why the conflict paid

Strip the settlement down to the incentive it targeted. Suppose, pre-2003, an analyst's bonus was informally tied to banking fees their coverage helped generate — say 1% of mandate fees in their sector. In a hot year their sector's IPOs and follow-ons raised \$5,000,000,000 in fees-bearing deals at a 5% fee:

$$
\\$5{,}000{,}000{,}000 \times 5\% = \\$250{,}000{,}000 \text{ in banking fees}, \quad 1\% = \\$2{,}500{,}000 \text{ to the analyst.}
$$

That \$2.5M dwarfs any salary, and *every dollar of it depended on companies wanting to hire the bank* — which depended on the analyst staying friendly. The settlement's core move was to cut that line entirely: an analyst could be paid for the *quality and impact* of research, but not as a share of the banking revenue their optimism attracted. **Intuition: you cannot regulate an analyst into honesty while their paycheck rises with every Buy that wins a mandate — so the fix was to break the paycheck link, not to police the prose.**

Did the settlement work? Partly. The most flagrant abuses — analysts attending banking pitches, bonuses tied to specific deals, bankers editing research — were stamped out, and the disclosure box made the conflicts visible. The \$450M for independent research seeded a small industry of conflict-free research boutiques. But the *softer* pressures survived intact, because the settlement could not legislate them away. Analysts still need corporate access, still hope their bank wins the mandate, still face the career asymmetry of the lone Sell. So the headline distribution barely budged: studies of post-settlement ratings found buys still dominating and sells still in the single digits. The settlement changed the *plumbing* of the conflict (no more direct pay link, full disclosure) without changing the *gradient* (optimism is still the path of least resistance). That is why a second, structurally different reform — MiFID II — came at the problem from the demand side a decade and a half later.

A useful way to see the limit of the 2003 reforms is to notice what they left untouched: the *buyer* of research. The settlement reformed how research was produced and disclosed, but the buy-side still got research "free," bundled into commissions. As long as research had no price, there was no market pressure for it to be good — a fund that received fifty mediocre Buy-skewed notes paid the same (zero, explicitly) as one that received five excellent ones. The 2003 reforms fixed the *integrity* of research without touching its *economics*. MiFID II did the opposite: it left integrity rules largely alone and attacked the economics, by forcing research to be bought and sold at an explicit price.

## IPO research quiet periods and the booster shot

The conflict is sharpest at the exact moment research matters most to a retail investor: a new IPO. The bank underwriting an IPO has every reason to want the stock to do well after it lists, because a successful deal wins future mandates and because the bank's own analysts cover it. So securities rules impose a **quiet period**: the underwriters' analysts may not publish research on the company for a window after the IPO (post-settlement rules set this at roughly 25–40 days depending on the rule and the bank's role; it has varied over time and the JOBS Act loosened it for smaller "emerging growth" companies).

The purpose is to stop the **booster shot** — the pattern where, the day the quiet period ends, the underwriting banks' analysts simultaneously initiate coverage with glowing Buy ratings and high targets, conveniently supporting a stock their own firm just sold to the public. Studies after the dot-com era documented exactly this: stocks tended to get a cluster of bullish initiations from their underwriters right as the quiet period lapsed, and those underwriter ratings were systematically more optimistic than independent analysts' on the same name.

The quiet period is a neat illustration of how disclosure-based regulation tries to manage a conflict it cannot eliminate. Regulators cannot forbid an underwriter's analyst from ever covering the company — that would leave the IPO with even less research, hurting investors. So instead they impose a *timing* rule: stay silent for a window, so the first burst of bullishness cannot ride the immediate post-IPO momentum and so that any initiation arrives after the market has had a chance to form its own view. It is a compromise, and like most compromises it is leaky. The "booster shot" still happens; it just happens on day 26 instead of day 2. And the JOBS Act of 2012, designed to make it easier for smaller "emerging growth companies" to go public, deliberately *loosened* these restrictions — allowing underwriter research sooner — on the theory that more research helps small issuers. The tension is permanent: every rule that suppresses conflicted research also suppresses *information*, and regulators are forever choosing how to trade one against the other.

Another structural feature amplifies the post-IPO risk: the **lock-up.** Insiders and early investors typically agree not to sell their shares for 90–180 days after the IPO. When the lock-up expires, a wave of insider selling can hit the stock — and it often coincides, awkwardly, with the period when underwriter research has turned bullish. An investor who bought on the booster-shot Buy can find themselves holding the bag exactly as the people who knew the company best are heading for the exit. The next worked example puts numbers on that trap.

#### Worked example: the cost of following a conflicted Buy into an IPO

A retail investor buys \$10,000 of a hot IPO. The stock prices at \$20 and pops to \$28 on day one — a 40% first-day return that the investor, buying at the open, mostly misses (they get filled around \$27). Twenty-five days later, the underwriters' analysts initiate with Buy ratings and \$40 targets, and the investor, reassured, holds. Over the next year the lock-up expires, early insiders sell, and the stock drifts to \$15.

The investor's loss, having bought ~370 shares at \$27:

$$
(\\$15 - \\$27) \times 370 \approx -\\$4{,}440, \text{ a } 44\% \text{ loss on } \\$10{,}000.
$$

The "Buy with a \$40 target" that arrived right after the quiet period was not independent advice — it came from the firms that earned the underwriting fee on the very shares the investor was holding. **Intuition: the research you most want to trust at an IPO comes from the party with the most reason to talk the stock up, which is why the first-day pop and the post-quiet-period booster shot are warning signs, not buy signals.**

![Bar chart of the US median first-day IPO return by year showing the pop](/imgs/blogs/sell-side-research-the-analyst-the-rating-and-the-wall-2.png)

That first-day pop is not free money for ordinary buyers — most of it accrues to the institutions allocated shares at the offer price, a point developed in the sibling post on [how the IPO price is set](/blog/trading/capital-markets/bookbuilding-and-price-discovery-how-the-ipo-price-is-set). The research that lands right afterward is part of the same machinery.

## MiFID II unbundling: pricing research separately

Europe attacked the conflict from a completely different angle. The problem there was not just optimism — it was that research was *bundled*. A buy-side fund paid its broker a single trading commission, and "free" research came along with it. Because research had no explicit price, two distortions followed: funds over-consumed research they barely read, and the cost was buried in commissions ultimately paid by the funds' own end investors.

**MiFID II** — the EU's Markets in Financial Instruments Directive II, in force **January 2018** — forced **unbundling**: asset managers had to pay for research *separately* from execution, either out of their own P&L or from a clearly disclosed, client-approved research budget. Research now had to carry a price tag.

![Before-and-after diagram of MiFID II unbundling research payment from trade execution](/imgs/blogs/sell-side-research-the-analyst-the-rating-and-the-wall-8.png)

The effects were large and mostly one-directional:

- **Budgets fell.** Once funds had to write a check for research instead of getting it "free," they bought far less of it. Estimates put the drop in European research spending at roughly 20–30% in the first years.
- **Coverage thinned, especially small caps.** Banks cut analysts and dropped coverage of companies that were not generating trading or banking revenue — disproportionately smaller firms. A small-cap company that loses all analyst coverage becomes harder for investors to find and value, which can widen its spreads and raise its cost of capital. This is the spine biting back: less information about a claim makes that claim harder to fund.
- **Quality bifurcated.** Top-tier research on mega-cap names stayed (funds will pay for it); marginal research on marginal names vanished.

#### Worked example: a MiFID II research budget

A European equity fund manages \$2,000,000,000. Pre-MiFID, it implicitly paid for research through commissions — invisibly. Post-MiFID, it sets an explicit annual research budget of, say, 4 basis points of assets:

$$
\\$2{,}000{,}000{,}000 \times 0.04\% = \\$800{,}000 \text{ per year for research.}
$$

Now every research provider must justify a slice of that \$800k. The fund ranks brokers by how much their research actually informs decisions, drops the bottom half, and concentrates spending on a few. The small bank whose research the fund "got for free" but rarely used now gets \$0 — and its analysts covering small caps are first to be cut. **Intuition: putting a price on research did exactly what prices do — it killed the demand for research nobody was really paying for, including a lot of small-cap coverage that, for all its conflicts, was the only information some companies had.**

The US did not adopt unbundling, and after MiFID II created a cross-border headache (US brokers giving research to EU funds suddenly looked like unregistered investment advisers being paid for advice), the SEC issued no-action relief to keep the two regimes compatible. The result is a transatlantic split: bundled in the US, unbundled in Europe. The relief lapsed and was a recurring source of friction; the practical upshot is that global asset managers run two different research-payment processes depending on which side of the Atlantic the money sits.

Tellingly, the unbundling consensus has since wobbled. By the early 2020s, European policymakers worried that MiFID II had gone too far — that the collapse in small- and mid-cap coverage was starving exactly the growth companies the EU wanted its capital markets to fund. Subsequent reform efforts began to *re-allow* limited bundling for smaller issuers, an implicit admission that the cure had a serious side effect. This is the deepest lesson of the whole episode: research is not a free-floating service that can be priced like any other. It is a semi-public good — once an analyst covers a company, the whole market benefits from the price discovery, even investors who never paid. Price it strictly and you under-produce it, exactly as economics predicts for public goods. Leave it free and bundled and you over-produce skewed, conflicted noise. There is no clean equilibrium, only a choice of which failure mode to tolerate.

## How the buy-side actually uses (and discounts) sell-side research

Here is the twist that ties it together: the professionals who receive sell-side research mostly *know* it is conflicted, and they use it accordingly. The buy-side does not read a "Buy with a \$100 target" as gospel. They use the sell side for a few specific things and discount the rest.

What the buy-side genuinely values:

- **The model and the data.** A good analyst's earnings model — the build of revenue by segment, the assumptions, the channel checks — is real work the buy-side would otherwise have to do itself. Funds often care more about the *spreadsheet* than the rating.
- **Access and conferences.** The single most valued service is **corporate access**: the analyst arranges the meeting between the fund's PM and the company's CFO. This is also why the sell side guards relationships with management so jealously.
- **The consensus benchmark.** What matters on earnings day is not whether a stock is "good" but whether it beats or misses *consensus*. Sell-side estimates *are* the consensus, so they define the bar even for investors who ignore the ratings.
- **A second opinion and a sounding board.** A sharp analyst who knows a sector cold is worth a phone call, regardless of what their published rating says.

What the buy-side discounts:

- **The rating itself**, which everyone knows is skewed. A sophisticated PM watches **rating changes** (an upgrade or downgrade carries information about a *change* in view) far more than rating *levels*.
- **The price target**, treated as the analyst's stated bull case, not a forecast.

This is the deepest point about research: its value to professionals is in the *raw material* — the models, the data, the access, the change in view — not the headline verdict. The verdict is for the unsophisticated, and the unsophisticated are the ones the conflict harms most. The sell-side/buy-side relationship is explored further in the sibling post on [who actually owns the market](/blog/trading/capital-markets/the-buy-side-who-actually-owns-the-market); research is one of the main currencies in that relationship.

![Line chart of US equity market capitalization by year, the claims research helps price](/imgs/blogs/sell-side-research-the-analyst-the-rating-and-the-wall-6.png)

The reason research exists at all is that there is a vast stock of claims — a US equity market worth tens of trillions of dollars — that someone has to price every day. Research is the labor of pricing those claims at scale.

## Common misconceptions

**"Banks sell research for a profit."** No. A research note is given to clients for free; the desk is usually a cost center. As the first worked example showed, a \$60M-a-year department earns its keep by feeding trading commissions and helping win underwriting mandates — a single \$35M IPO fee covers more than half its cost. The product is the loss leader; the bank's real products are execution and underwriting.

**"A Buy rating means the analyst thinks you should buy."** Often, but the rating is the most skewed and least informative part of the note. Roughly half the market is rated Buy and only single digits Sell, so a Buy carries little signal. The model, the estimate revision, and especially a *change* in rating carry far more.

**"A high price target is a forecast."** A target is the output of a valuation under the analyst's assumptions, framed to advertise upside. As shown, the band between a Buy and a Hold can be a few percent of implied return — and the incentives all push the target up. Treat it as a stated bull case.

**"The Chinese wall is just for show."** The wall is genuinely effective at its actual job — stopping the *crime* of trading on material non-public information, via watch lists, restricted lists, system separation, and logged wall-crossings. What it never policed was the *legal* conflict of writing optimistically to please a prospective client; that took the 2003 settlement, and even then only by cutting the analyst's pay link to banking.

**"MiFID II fixed research."** It fixed the *bundling* distortion by putting a price on research — but the price did what prices do: it cut demand. Budgets fell ~20–30% and small-cap coverage thinned, leaving some companies with no analyst following at all. Solving the conflict cost the market information.

## How it shows up in real markets

**The 2003 settlement and the disclosure box.** Open almost any sell-side equity report today and you will find, near the back, a dense disclosures section: the bank's banking relationship with the company, its ownership, and a bar showing the firm-wide percentage of Buy/Hold/Sell ratings and what share of each rating bucket are also banking clients. That box is a direct, lasting artifact of the 2003 Global Analyst Research Settlement — a structural memory of Blodget and Grubman embedded in every report.

**The 2021 IPO boom and the 2022 bust.** 2021 was a record year — about \$142bn of US IPO proceeds across ~397 deals, with first-day pops averaging in the 30s of percent. Then 2022 collapsed to roughly \$8bn across 71 deals as rates rose. Many of the 2021 listings that got bullish underwriter initiations right after their quiet periods fell 60–80% over the following year. The pattern the booster-shot rule was written to catch played out in plain sight: the research that supported the deals came overwhelmingly from the banks that earned the fees.

![Bar chart of US IPO proceeds by year showing the 2021 peak and 2022 collapse](/imgs/blogs/sell-side-research-the-analyst-the-rating-and-the-wall-4.png)

**The MiFID II coverage cliff for small caps.** After January 2018, surveys found a measurable share of European small- and mid-cap companies lost *all* analyst coverage. A company with zero research is invisible to the screening tools many funds use, which can widen its bid-ask spread and shrink its investor base. This is the clearest real-market proof that research is part of the plumbing of capital formation, not a luxury bolted on top.

![Horizontal bar chart of bid-ask spread by liquidity tier showing wide spreads for small caps](/imgs/blogs/sell-side-research-the-analyst-the-rating-and-the-wall-7.png)

The spread-by-tier picture shows why losing coverage hurts: micro-caps already trade at spreads near 80 basis points versus ~1 bp for a mega-cap. Take away the one or two analysts who follow a small company and you remove the information that might have narrowed that spread — raising its cost of capital exactly when it can least afford it.

**Credit research and the rating-agency cousin.** Sell-side *credit* research carries the same skew in a quieter way, and it sits next to a separate but related institution: the credit-rating agencies (Moody's, S&P, Fitch), whose conflicts in structured products were central to 2008. The agencies are part of the [supporting cast of exchanges, data, indices, and ratings](/blog/trading/capital-markets/the-supporting-cast-exchanges-data-indices-and-ratings) — paid by the issuer, just as sell-side research is funded by the side that wants the deal to happen.

## The takeaway: read research for its raw material, not its verdict

Sell-side research is the information layer of the capital-markets machine — the labor of pricing thousands of claims so that savers can decide which firms to fund. But it is produced by intermediaries whose money comes from trading those claims and underwriting new ones, so it tilts optimistic by construction. Three waves of reform — the Chinese wall, the 2003 settlement, and MiFID II — each addressed a different slice of the conflict, and each had a cost: the settlement broke the pay link but left the access and relationship pressures; unbundling priced research honestly but starved small caps of coverage.

The practical lesson maps cleanly onto how the best buy-side investors behave. **Use research for its raw material — the model, the data, the corporate access, and above all the *change* in an analyst's view — and discount the headline rating and target, which are the most conflicted and least informative parts.** A Buy is nearly the default; a downgrade from Buy to Hold by a respected analyst is real information; a rare outright Sell is a klaxon.

And the spine-level point: research is what lets the secondary market price a security accurately enough that the primary market can keep issuing new ones. When research thins — as it did for European small caps after MiFID II — issuance gets harder for exactly the companies that most need outside capital. The information layer is not decoration on the machine. It is part of how money finds its best use, and when it bends to the people who pay for it, capital flows to the wrong place. The whole point of disclosure-based regulation and the wall is to keep that bend within bounds — never to pretend the bend isn't there.

## Further reading & cross-links

- [Inside an investment bank: ECM, DCM, M&A, and trading](/blog/trading/capital-markets/inside-an-investment-bank-ecm-dcm-ma-and-trading) — where the research desk sits relative to the private banking side.
- [The buy-side: who actually owns the market](/blog/trading/capital-markets/the-buy-side-who-actually-owns-the-market) — the consumers of research and how they discount it.
- [Bookbuilding and price discovery: how the IPO price is set](/blog/trading/capital-markets/bookbuilding-and-price-discovery-how-the-ipo-price-is-set) — the deal moment where research conflicts bite hardest.
- [The supporting cast: exchanges, data, indices, and ratings](/blog/trading/capital-markets/the-supporting-cast-exchanges-data-indices-and-ratings) — the rating agencies, research's issuer-paid cousins.
- [Inside an investment bank: how they make money](/blog/trading/finance/inside-an-investment-bank-how-they-make-money) — the fee economics that fund the research desk.
