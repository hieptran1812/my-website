---
title: "Corporate governance, fiduciary duty, and shareholder activism"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "How corporate law decides who controls a company and for whose benefit, why that control structure creates or destroys shareholder value, and how to read governance to find value an activist can unlock."
tags: ["corporate-governance", "fiduciary-duty", "shareholder-activism", "13d", "poison-pill", "dual-class", "proxy-fight", "delaware", "control-premium", "regulation"]
category: "trading"
subcategory: "Law & Geopolitics"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — Corporate law decides *who controls a company* and *for whose benefit*, and that control structure is itself a price: read the governance and you can spot value trapped behind it.
>
> - The shareholders own the firm but the **board controls** it; the board owes two **fiduciary duties** — care and loyalty — and the **business-judgment rule** shields its honest, informed decisions from courts. Most large US firms incorporate in **Delaware** because that body of law is the most developed.
> - Activists exploit the gap between price and value: build a stake, file the **Schedule 13D** at 5%, and push for a buyback, a breakup, a sale, or board seats. The 13D itself usually pops the stock 5–10% in a day, before anything has actually changed.
> - Boards resist with **defenses** — the poison pill, the staggered board, dual-class super-voting shares. Each blunts the activist and acquirer, and the deeper the entrenchment the deeper the **governance discount** the stock tends to carry.
> - The one number to remember: a control structure that caps a 15x-multiple business at 10x is suppressing **a third of its value** (\$50 vs \$75 on \$5 of EPS). Lifting the overhang re-rates the same earnings — that re-rating, plus the **control premium** an acquirer pays (typically **20–40%** over the unaffected price), is the entire activist trade.

On January 22, 2013, Carl Icahn disclosed a large stake in Dell and a campaign against Michael Dell's plan to take the company private at \$13.65 a share. He argued the price was a steal — that the board, led by a founder who would be on *both* sides of the deal, was selling the company to its own CEO too cheaply. Over the next eight months the fight played out in proxy filings, in Delaware court, and in the financial press. When the dust settled, the buyout group raised its offer and added a special dividend, pushing the effective price to roughly \$13.96 plus a \$0.13 dividend. The raise was a few percent — but on a company valued near \$25 billion, a few percent is several hundred million dollars that went to public shareholders instead of staying with the buyer. The fight was not about the business. The business never changed. It was about *who controlled the sale process and for whose benefit* — and that is a question of corporate law.

This is the recurring shape of the most lucrative trades in equity markets: a business whose *operations* are fine but whose *control structure* is mispriced. A founder with super-voting shares running the company for his vision rather than the float. A board so entrenched it has stopped feeling the cost of capital. A conglomerate worth more in pieces than as a whole, kept together because no one with the votes wants to break it up. In each case there is value sitting between what the assets are worth and what the stock trades at, and the bridge across that gap is built from a handful of legal instruments — fiduciary duties, the business-judgment rule, the poison pill, the 13D, the proxy fight, the Revlon and Unocal standards. Learn to read those, and you can see the trapped value before the campaign that releases it.

This post sits inside a series on how law moves markets. Where [regulatory risk as an asset-pricing factor](/blog/trading/law-and-geopolitics/regulatory-risk-as-an-asset-pricing-factor) shows how *external* rules (antitrust, a banned product) get priced into a discount rate, this post is about the *internal* rulebook — the corporate-law machinery that decides who runs the firm. It is the legal substrate beneath [merger arbitrage and regulatory deal risk](/blog/trading/law-and-geopolitics/merger-arbitrage-trading-regulatory-deal-risk), and it leans on the disclosure regime we cover in [disclosure and accounting law: SOX, IFRS vs GAAP](/blog/trading/law-and-geopolitics/disclosure-and-accounting-law-sox-ifrs-vs-gaap). By the end you will be able to map a firm's governance, size the value a control structure traps, and judge whether an activist can realistically pry it loose.

![Flow from shareholders electing the board through fiduciary duties and entrenchment defenses to value trapped as a governance discount versus value unlocked by activist pressure](/imgs/blogs/corporate-governance-fiduciary-duty-and-shareholder-activism-1.png)

## Foundations: the corporation, the board, and who controls it

Before any of the trades make sense, you have to be precise about a deceptively simple question: who owns a company, and who *runs* it? Almost everyone gets this wrong, and the gap between the two is where the whole discipline lives.

### The corporation is a nexus of contracts

Start from zero. A **corporation** is a legal fiction — an artificial "person" the law creates so that a group of people can pool capital, own assets, sign contracts, and be sued, all under one name, with the owners' personal assets protected (this is **limited liability**: if the firm fails, you lose your investment but not your house). Economists describe the corporation as a **nexus of contracts** — a central hub that ties together a web of agreements between everyone who deals with the firm: shareholders who put up equity capital, lenders who put up debt, employees who supply labor, suppliers, customers, and managers who run the operation day to day. The corporation is the knot where all those contracts meet.

The reason this framing matters for an investor is that it tells you the corporation has *no single owner-operator*. It is a bundle of claims. The shareholders hold the **residual claim** — they get whatever is left after everyone else (lenders, employees, suppliers, the tax authority) has been paid. That makes them the owners in the economic sense: they bear the upside and the downside, and they are last in line. But being last in line and bearing the risk does *not* mean they run the company. They almost never do.

### The separation of ownership and control

Here is the single most important fact in corporate governance, and the one most investors misstate: **the shareholders own the firm, but the board of directors controls it.** This is the *separation of ownership and control*, first named by Adolf Berle and Gardiner Means in 1932, and it is the root of everything else in this post.

Shareholders do not vote on whether to launch a product, fire an executive, or buy a rival. They elect a **board of directors** — a small group, often nine to twelve people — and the board hires, monitors, and (when necessary) fires the CEO, sets strategy at a high level, and approves the big moves: mergers, large issuances, dividends, the sale of the company. The board, in turn, delegates day-to-day management to the executives. So the chain of control runs: shareholders → elect → board → hires/oversees → management → runs the firm. The shareholders sit at the top of the legitimacy chain and the bottom of the cash-flow waterfall, but in between, the *board* is the seat of power.

This separation creates the **principal–agent problem**: the shareholders (principals) want the firm run to maximize the value of their residual claim, but the board and management (agents) are human beings with their own incentives — empire-building, comfortable lives, pet projects, self-preservation. Governance is the set of legal and structural devices that try to keep the agents working for the principals. When those devices are strong, the gap between price and value is small. When they are weak — an entrenched board, a founder who answers to no one — the gap can be enormous, and that gap is the activist's hunting ground.

The cost of that gap has a name: **agency cost**. Economists Michael Jensen and William Meckling formalized it in 1976 as the sum of three things: the money owners spend *monitoring* the agents (auditors, boards, proxy advisers), the money agents spend *bonding* themselves to the owners' interest (performance pay, covenants), and the **residual loss** — the value still destroyed despite both, because no contract is perfect. Every governance device in this post is an attempt to shrink one of those three components. A poison pill is a monitoring/control tool the board wields; performance-linked pay is a bonding device; and the *residual loss* — the empire-building acquisition, the hoarded cash earning nothing, the division kept for prestige — is exactly the value an activist tries to recover. When you size a governance discount in dollars, you are, in effect, estimating the market's verdict on a firm's agency cost. A firm priced at a wide discount to its unlockable value is a firm the market believes has high, persistent agency cost; the activist's bet is that the cost can be forced down.

### Fiduciary duties: care and loyalty

To bridge the principal–agent gap, the law imposes **fiduciary duties** on directors. A fiduciary is someone legally required to act in another's interest rather than their own; a trustee, a guardian, and a corporate director are all fiduciaries. Directors owe their duties to the corporation and its shareholders, and there are two of them.

The **duty of care** requires a director to be *informed* and *diligent* — to do the homework before deciding. A director who rubber-stamps a merger after a two-hour meeting without reading the materials or getting a valuation has breached the duty of care. The famous case is *Smith v. Van Gorkom* (Delaware, 1985), where the court held directors personally liable for approving a buyout in a single meeting without adequate information. The duty of care is about *process*: did you take the steps a careful person would take?

The **duty of loyalty** requires a director to put the corporation's interest ahead of their own — *no self-dealing*. A director who steers a contract to a company they secretly own, or who structures a deal to enrich themselves at shareholders' expense, has breached the duty of loyalty. Loyalty is the duty that bites hardest, because the courts will not give a self-dealing director the benefit of the doubt the way they will give a merely careless one. When a CEO tries to buy the company they run — the Dell situation — the duty of loyalty is squarely in play, because the CEO is on both sides.

![Board owes a duty of care and a duty of loyalty, with the business-judgment rule shielding informed decisions and enhanced review applying to conflicts and takeovers](/imgs/blogs/corporate-governance-fiduciary-duty-and-shareholder-activism-2.png)

### The business-judgment rule: why boards usually win

If directors can be sued for every decision that turns out badly, no competent person would ever serve on a board. So the law gives directors a powerful shield: the **business-judgment rule**. It is a *presumption* that, when directors make a decision on an informed basis, in good faith, and in the honest belief that it serves the company, a court will **not** second-guess the substance of that decision — even if it was, in hindsight, a terrible one.

The business-judgment rule is the reason most shareholder lawsuits against boards fail. A board that closes a factory, overpays for an acquisition, or rejects a buyout offer is almost always protected, *as long as the process was sound*: they were informed, they had no conflict, and they acted in good faith. The rule shifts the question from "was this the right decision?" (which courts refuse to answer) to "was this an honest, informed decision by disinterested directors?" (which courts will police). This is why activist campaigns rarely win in court on the *merits* of a strategy — courts defer to the board's judgment — and instead win on *process* (the board was conflicted or uninformed) or, far more often, in the *court of public opinion and the proxy vote*, where shareholders can simply replace the board.

There are two big exceptions where the business-judgment rule's deference is replaced by tougher **enhanced scrutiny**, both involving takeovers. Under the **Unocal standard** (*Unocal v. Mesa*, 1985), when a board adopts a *defensive* measure against a takeover, it must show the threat was real and the response was *proportionate* — because a board defending against a takeover is inherently conflicted (it is defending its own jobs). Under the **Revlon standard** (*Revlon v. MacAndrews & Forbes*, 1986), once a sale or break-up of the company is *inevitable*, the board's duty narrows to a single goal: get the **highest price** for shareholders. "Revlon mode" turns the directors into auctioneers; they can no longer favor one bidder for strategic reasons, only for price. These two standards are the legal levers an activist or rival bidder pulls to attack a board's defenses, and we will return to them.

### Delaware: why the rules live in one small state

One more foundation. When you read that a company is "a Delaware corporation," it usually has nothing to do with where it operates. Roughly **two-thirds of Fortune 500 companies** and the majority of US public firms are incorporated in **Delaware**, a state of barely a million people. They choose Delaware because corporate disputes there are heard by the **Court of Chancery** — a specialized business court with no juries, expert judges, and, crucially, *the deepest body of case law in the world on exactly these questions*. When a board wonders whether a poison pill will survive a challenge, or whether they are in Revlon mode, there is almost certainly a Delaware precedent that answers it.

For an investor, "Delaware law governs" is a feature: it means the rules of the fight are knowable in advance. The Unocal and Revlon standards, the business-judgment rule, the validity of a poison pill — these are Delaware doctrines, and they apply to most of the large US companies you will ever analyze. (A handful of firms have recently flirted with reincorporating elsewhere when they felt Delaware courts turned against insiders — a 2024 ruling voiding a multi-billion-dollar pay package was a flashpoint — but Delaware remains the overwhelming default.) The practical takeaway: when you assess a governance situation, you are usually assessing it against Delaware law, and that law is unusually predictable.

### Shareholder rights: the vote is the lever

Against the board's control, what do shareholders actually have? One real lever and a few smaller ones, all flowing from the **vote**. Each share of common stock normally carries one vote. Shareholders vote at the **annual meeting** on the election of directors, on certain major transactions (a merger, a charter amendment), and — since the 2010 Dodd-Frank Act — on an advisory **say-on-pay** vote on executive compensation. Because most shareholders do not attend in person, they vote by **proxy**: they authorize someone to cast their votes, and the contest to win those authorizations is a **proxy fight**.

The vote is the activist's ultimate weapon, because it is the one thing the board cannot ignore. A board can refuse to meet an activist, reject every proposal, and hide behind the business-judgment rule — but it cannot stop shareholders from voting it out at the annual meeting. That is why the structures that *neutralize the vote* — the staggered board that lets only a third of seats change each year, the dual-class shares that give a founder permanent voting control — are the most powerful defenses of all, and the ones that create the deepest discounts. To understand the trades, we now have to understand the offense (the activist playbook) and the defense (the entrenchment toolkit), and how they price against each other.

## The activist playbook: building a stake, the 13D, and the value to unlock

A **shareholder activist** is an investor who buys a meaningful stake in a company and then *agitates for change* to make the stock worth more, rather than passively waiting. The classic activists — Icahn, Elliott Management, Third Point (Dan Loeb), Trian (Nelson Peltz), Starboard Value, ValueAct — run a recognizable playbook, and every step has a legal trigger.

### Step one: build a stake quietly

The campaign begins with accumulation. The activist buys shares in the open market, deliberately staying *below the disclosure threshold* so the market does not see them coming and bid the price up. In the US, that threshold is **5% of a company's voting shares**. As long as you hold under 5%, you can accumulate without telling anyone.

### Step two: cross 5% and file the Schedule 13D

The moment an investor crosses **5%** with an intent to influence the company, US securities law (the Williams Act, Section 13(d) of the 1934 Exchange Act) requires them to file a **Schedule 13D** with the SEC. As of 2024 the deadline is **five business days** after crossing 5% (it was historically ten calendar days — the SEC tightened it to give the market faster information). The 13D is a public document, and it must disclose *who* is buying, *how much* they hold, *how* they paid for it, and — the part that moves the stock — their **purpose**: do they intend to push for board seats, a sale, a spin-off, a buyback, a strategy change?

The 13D is the starting gun. It tells the entire market that a known activist now owns a chunk of the company and intends to fight. (There is a passive cousin, the **Schedule 13G**, for index funds and others who cross 5% with no intent to influence — that one does *not* move the stock, because it signals no campaign.) Reading 13D filings is itself a strategy: when a respected activist files, others pile in, both because the activist may force a value-unlocking change and because the activist's mere presence raises the odds of a takeover.

![Activist campaign pipeline from quietly building a 5 to 10 percent stake through filing the Schedule 13D and the announcement pop to engaging the board, a proxy fight, and a value-unlocking outcome](/imgs/blogs/corporate-governance-fiduciary-duty-and-shareholder-activism-3.png)

### Step three: the announcement pop

Empirically, the 13D filing itself moves the stock. Academic studies of US activist campaigns (the canonical work is Brav, Jiang, Partnoy, and Thomas) find an average **abnormal return of roughly 6–7% around the 13D filing window**, with no reversal over the following year — the market reprices the stock *up* on the news that an activist is involved, and on average that gain sticks. Why does a piece of paper raise the value of a business that has not changed? Because the filing changes the *probability distribution of outcomes*: it raises the odds of a buyback, a breakup, better capital discipline, or a sale at a premium. The market is pricing the expected value of the campaign before the campaign has done anything.

#### Worked example: the 13D-announcement pop

Consider a company with **200 million shares** trading at **\$40**, for a market capitalization of **\$8.0 billion**. A well-known activist files a 13D disclosing a 6% stake and a plan to push for a large buyback and two board seats.

- Suppose the stock pops **7%** on the filing — the historical average for a credible campaign. New price = \$40 × 1.07 = **\$42.80**.
- Market-cap change = 200 million × (\$42.80 − \$40.00) = 200,000,000 × \$2.80 = **\$560 million** of value created in a day.
- The activist's own 6% stake (12 million shares) just gained 12,000,000 × \$2.80 = **\$33.6 million** on paper, before the campaign has produced a single concrete change.

The takeaway: the 13D pop is the market repricing the *expected* outcome of the fight — a \$560 million swing in value bought with nothing but a public filing and a reputation.

### Step four: engage, then escalate to a proxy fight

After the 13D, the activist engages the board — privately at first (letters, meetings), then publicly (open letters, presentations, a dedicated campaign website) if the board resists. The demands are usually drawn from a short menu, each of which can genuinely create value:

- **Return cash**: a large **buyback** or special dividend, on the theory that the company is hoarding cash that earns nothing and should be returned to shareholders (who can redeploy it at a higher return). A buyback also raises earnings per share by shrinking the share count.
- **Break up the company**: a **spin-off** of a division, on the theory that the parts are worth more separately than together — the **conglomerate discount**, which we work through below.
- **Sell the company**: put it up for sale to a strategic or private-equity buyer who will pay a **control premium**.
- **Cut costs / fix capital allocation**: improve margins, stop value-destroying acquisitions, refresh management.
- **Board seats**: the structural demand that backs all the others — put the activist's nominees on the board so they have a vote and a voice from the inside.

If the board refuses, the activist escalates to a **proxy fight**: it nominates its own slate of directors and campaigns for shareholder votes at the annual meeting to *replace* some or all of the incumbents. This is the nuclear option, and it is expensive and uncertain — but the *threat* of it is what gives the activist leverage. Most campaigns settle: the board gives the activist a board seat or two, agrees to some of the demands, and avoids a public vote it might lose. The 2013–2015 wave of "settlements" became so routine that getting an activist a seat without a vote is now the default outcome of a credible campaign.

### The value an activist can unlock: a buyback and a breakup

The reason any of this works is that there is *real value* to release — the activist is not conjuring it from thin air. The two largest sources are returning excess capital and undoing a conglomerate discount.

#### Worked example: the value a buyback unlocks

Take a company with **\$10 billion** market cap, **500 million shares** at **\$20**, earning **\$1.00 per share** (so a 20x P/E, \$500 million of net income). It sits on **\$2 billion** of excess cash earning almost nothing. An activist pushes the board to use that cash to buy back stock.

- At \$20 a share, \$2 billion buys back \$2,000m / \$20 = **100 million shares**, cutting the count from 500m to **400m**.
- Net income is roughly unchanged at \$500 million (the cash was earning almost nothing, so losing its trivial interest barely dents earnings).
- New EPS = \$500m / 400m shares = **\$1.25** — up from \$1.00, a **25% jump** in EPS.
- If the market holds the same 20x multiple, the price goes to 20 × \$1.25 = **\$25**, a **25% gain** from \$20.

The cash did nothing on the balance sheet; redeployed into a buyback it lifted EPS 25% and, at a constant multiple, the stock 25%. That mechanical accretion is why "return the cash" is the single most common activist demand. (The same lever, run across the whole index, is why post-2017 tax reform unleashed a buyback record — see [the 2017 TCJA and the repatriation trade](/blog/trading/law-and-geopolitics/the-2017-tcja-and-the-repatriation-trade).)

The buyback is the lever activists push more than any other, and at the index level it is enormous: S&P 500 companies have repurchased roughly **\$500–950 billion of their own stock every year**, with a record surge after the 2017 corporate-tax cut freed up cash.

![S&P 500 gross buybacks per year in US dollar billions from 2016 to 2024 with the post-2017 record years highlighted](/imgs/blogs/corporate-governance-fiduciary-duty-and-shareholder-activism-7.png)

The second great unlock is the **sum-of-the-parts** breakup. A **conglomerate** — a company spanning several unrelated businesses — often trades at a discount to what its divisions would be worth as separate, focused companies. Investors who want clean exposure to one business do not want to buy the whole messy bundle; analysts cannot cover it well; capital gets misallocated across divisions. The activist's move is to value each segment on the multiple a *pure-play peer* in that industry commands, sum them, and show that the total exceeds the conglomerate's market cap. The gap is the **conglomerate discount**, and a spin-off can release it.

#### Worked example: the sum-of-the-parts breakup

A conglomerate trades at a **\$75 billion** market cap. It has three segments. Valued separately on peer multiples:

- **Segment A** (a stable cash cow): \$3.3B of EBITDA at a peer 12x = roughly **\$40 billion**.
- **Segment B** (a fast grower): \$1.4B of EBITDA at a peer 25x = roughly **\$35 billion**.
- **Segment C** (a real-estate / asset-heavy unit): appraised at roughly **\$25 billion**.
- Sum of the parts = \$40B + \$35B + \$25B = **\$100 billion**.

The pieces are worth \$100 billion; the whole trades at \$75 billion. The **conglomerate discount is \$25 billion**, or 25% of the standalone value. If an activist forces a spin-off that lets each segment re-rate to its peer multiple, shareholders capture roughly \$100B − \$75B = **\$25 billion**, a **+33% gain** on the \$75 billion they held. That is the breakup thesis in one number.

![Sum-of-the-parts comparison showing three segments worth 100 billion versus a 75 billion conglomerate market cap with a 25 billion breakup unlock](/imgs/blogs/corporate-governance-fiduciary-duty-and-shareholder-activism-6.png)

The catch — and it is always the catch in this discipline — is that the value only unlocks if the activist can actually *force* the change. A board with no defenses and a one-share-one-vote structure is exposed; a board armored with a poison pill, a staggered structure, and a founder's super-voting shares may sit on that \$25 billion indefinitely. The defenses are the other half of the picture.

## The defenses: how boards suppress the value

A board that does not want to be pushed around has a well-developed legal toolkit to resist activists and acquirers. Each defense raises the cost or lowers the odds of a hostile change of control — and each, by making the company harder to fix or take over, tends to *lower* the price the market will pay. The defenses are simultaneously the board's shield and the source of the governance discount.

![Matrix of takeover defenses showing the trigger, the effect on an activist, and the value effect for the poison pill, the staggered board, and dual-class shares](/imgs/blogs/corporate-governance-fiduciary-duty-and-shareholder-activism-4.png)

### The poison pill (shareholder-rights plan)

The **poison pill**, formally a *shareholder-rights plan*, is the most important takeover defense ever invented (by lawyer Martin Lipton in 1982). It works like this: the plan gives every shareholder *except* a hostile acquirer the right to buy new shares at a steep discount the moment that acquirer crosses a trigger threshold — typically **10% to 20%** of the stock. If the raider crosses the line, the pill "flips in": everyone else gets to buy cheap stock, **massively diluting the raider's stake** and making it prohibitively expensive to accumulate control. No acquirer wants to be diluted into oblivion, so in practice the pill *prevents* anyone from crossing the threshold without the board's permission. It does not stop a deal the board *wants*; it stops one the board *opposes*.

Crucially, a pill is almost always **redeemable** — the board can switch it off. So the pill does not block a takeover outright; it *forces the acquirer to negotiate with the board*, which is exactly the point. The board says, in effect, "you cannot buy this company over our heads; you must come through us." Delaware courts have upheld pills under the Unocal standard as long as they are a proportionate response to a real threat. Because the board *can* redeem the pill, its discount is the *shallowest* of the major defenses: an activist who wins board seats can vote to pull the pill, and the overhang lifts.

### The staggered (classified) board

A **staggered board** (or *classified board*) divides directors into classes — usually three — so that **only one class stands for election each year**. The consequence is brutal for an activist: even if you win every contested seat at one annual meeting, you can only replace *one-third* of the board. To win majority control you must win proxy fights in **two consecutive years**. That delay — two full annual cycles — is often enough to exhaust an activist's patience, capital, and investor base.

Combined with a poison pill, a staggered board is a formidable wall: the pill stops you from buying control, and the staggered board stops you from *voting* your way to control quickly. This "pill + staggered board" pairing was the gold standard of anti-takeover armor for decades. Its discount is *deeper and stickier* than a bare pill's, because there is no quick way to dislodge it — you cannot simply win one vote and pull the pill; you have to grind through two election cycles. (The trend among large-cap firms has been to *de-stagger* — annual election of all directors — under pressure from index funds and proxy advisers, which is itself a re-rating catalyst when it happens.)

### Dual-class shares: the founder's permanent control

The deepest entrenchment of all is the **dual-class share structure**. The company issues two classes of common stock: **Class A** shares sold to the public, carrying one vote each (or sometimes none), and **Class B** shares held by the founders and insiders, carrying **multiple votes each** — often 10, sometimes 20. The economics can be split fifty-fifty while the *votes* are split ninety-ten. This lets a founder own a minority of the *cash flow* while retaining majority *control* of the company — permanently, regardless of how many shares the public buys.

Dual-class is everywhere in modern technology: Alphabet (Google), Meta, Snap (whose IPO shares had *zero* votes), Ford, Berkshire Hathaway, and many others. The founder's pitch is that super-voting shares let them run the company for a long-term vision without being whipsawed by short-term shareholders or raiders — and for a Bezos, a Zuckerberg, or a Buffett, that argument has paid off spectacularly. But the structure also makes the company **effectively immune to activism and to hostile takeover**: you cannot win a proxy fight you can never out-vote, and you cannot buy control that is not for sale. An entrenched founder with super-voting stock answers to no one, and if that founder's judgment deteriorates, public shareholders have no remedy.

That immunity is why dual-class names tend to carry the *deepest and stickiest* governance discount — and why major index providers (S&P Dow Jones, FTSE Russell) and proxy advisers have pushed back, in some cases barring no-vote shares from flagship indexes. The discount is real but the verdict is not one-sided: the best founders justify it, and the worst destroy value behind it. That tension is the heart of the dual-class debate, which we take up in the misconceptions below.

### How the defenses become a discount

Put the offense and defense together and the pricing logic is clear. Value sits trapped between a stock's price and its intrinsic worth; an activist could release it; the defenses determine *whether the activist can*. The market is not stupid — it prices in the *probability* that the value gets unlocked. A clean-governance firm with no pill, no staggered board, and one-share-one-vote trades close to its unlockable value, because anyone can force the issue. A triple-armored firm — pill, staggered board, founder super-vote — trades at a deep discount, because the market correctly judges the trapped value may *never* be released. The defense suppresses the price precisely because it suppresses the *path* to the value.

#### Worked example: the governance discount and its re-rating

Two companies each earn **\$5.00 per share** and have identical businesses. The only difference is governance.

- **Firm A (entrenched)**: dual-class, staggered board, a pill in place. The market, judging that no one can ever force a change, prices it at a **10x** P/E. Price = 10 × \$5.00 = **\$50**.
- **Firm B (clean)**: one-share-one-vote, annual elections, no pill — peers with this governance trade at **15x**. Price = 15 × \$5.00 = **\$75**.

The entrenched firm is suppressing **\$25 of value per share** — \$75 of unlockable worth priced at \$50 — purely because of its control structure. That is a **33% discount** to the clean peer, on identical earnings. Now suppose the founder retires, the dual-class shares convert to one-vote, and the board de-staggers: the overhang lifts and the stock re-rates toward 15x. The move from \$50 to \$75 is a **+50% gain** on the same \$5 of EPS — the entire governance discount, released. The cash flows never changed; only who controls them, and on what terms, did.

![Before and after comparison of an entrenched firm at 10 times earnings priced at 50 dollars re-rating to a clean peer at 15 times earnings priced at 75 dollars on identical EPS](/imgs/blogs/corporate-governance-fiduciary-duty-and-shareholder-activism-5.png)

Notice the asymmetry between the two worked examples above. The discount is a *33% haircut* off the high price (\$25 off \$75); the re-rating is a *50% gain* off the low price (\$25 on \$50). Same \$25 of value, two different percentages, because the base is different — the way down is measured against the unlocked value, the way up against the suppressed price. This is the arithmetic of every value-trap trade, and it is why the upside on a successful governance unlock so often beats the downside if the overhang merely persists.

### Say-on-pay, ESG votes, and the universal proxy

Two more pieces of modern governance machinery shape these fights. The first is **say-on-pay**: since Dodd-Frank, public companies must hold a (usually non-binding) shareholder vote on executive compensation. A *failed* say-on-pay vote — a majority of shareholders rejecting the pay plan — does not legally force a change, but it is a public humiliation and a signal of weak board oversight; it often presages an activist campaign, because it advertises that shareholders are unhappy and the board is not listening. Alongside it, **ESG** (environmental, social, governance) proposals and the rise of proxy advisers (ISS, Glass Lewis) mean boards now face organized scrutiny on issues from climate disclosure to board diversity to executive pay — and the *governance* leg of ESG overlaps directly with everything in this post.

The second is the **universal proxy card**, mandatory in US contested elections since 2022. Before it, in a proxy fight, a shareholder voting by proxy had to choose *one side's card* — all incumbents or all challengers — and could not mix. The universal card lists *all* nominees, incumbents and challengers, on a single ballot, so shareholders can pick the best individuals from each side. This **lowered the cost of running a partial slate** and tilted the field toward activists: you no longer have to convince shareholders to throw out the whole board to get a couple of your people on it. The universal proxy is a quiet but real change in the balance of power, and it has made minority-slate campaigns more common and more winnable.

### The proxy-adviser ecosystem and the index funds

There is one more set of actors that decides who wins these fights, and it is invisible until you look for it: the **proxy advisers** and the **index funds**. Most shares of a large US company are not held by the founder or by activists — they are held by institutions, and a huge and growing slice sits in *passive* index funds run by a handful of giant managers (the largest three together vote a double-digit percentage of the entire S&P 500). Those managers cannot research every vote at every company they hold, so they lean heavily on two **proxy advisory firms** — Institutional Shareholder Services (ISS) and Glass Lewis — that publish voting recommendations on every contested item. When ISS recommends *for* an activist's slate, a large bloc of institutional votes tends to follow; when it recommends *against*, the campaign is usually dead. This is why activists spend so much effort crafting a thesis that will *win the proxy advisers* — the public letters and presentations are aimed as much at ISS and Glass Lewis as at the company.

For the practitioner, the proxy-adviser ecosystem is a readable signal. A proxy adviser recommendation in favor of an activist, or against a board's pay plan, materially raises the odds a campaign succeeds — and the recommendations are public, dated, and ahead of the vote. The index funds' published voting guidelines (on staggered boards, dual-class shares, over-boarded directors) are also public, and they tell you in advance which governance features the largest shareholders will *vote against* given the chance. Reading those guidelines is reading the medium-term pressure on a company's defenses: a structure the big index funds oppose is a structure under slow, persistent erosion, and the erosion is itself a re-rating catalyst.

### The control premium and the takeover defense

The last concept knits the offense and defense together: the **control premium**. Control of a company is worth *more* than a passive minority stake, because whoever has control can change strategy, replace management, capture synergies, and direct the cash flows. So an acquirer buying *control* of a company will pay more than the prevailing market price — historically a premium of **20% to 40%** over the **unaffected price** (the price before any rumor or bid leaked). That premium is the market value of control, made explicit.

Here is the interaction that ties the whole post together: a *takeover defense suppresses the realizable control premium*. If a board with a pill and a staggered board refuses to engage, an acquirer cannot capture control, so it will not pay the premium — and the stock, denied that bid, trades at the unaffected price or below. Remove the defenses (or put the company in "Revlon mode" via a sale process) and the control premium becomes capturable, so the price rises toward it. The defense and the premium are two sides of one coin: the defense is the *amount of control premium the board can withhold from shareholders*.

#### Worked example: the control premium in a takeout

A company's stock has been drifting at an **unaffected price of \$50** before any bid. An activist's 13D pops it to \$55 (a +10% move on the campaign expectation). Then a strategic acquirer launches a takeover, and after a brief auction the board — now in Revlon mode, legally obligated to maximize price — accepts a cash offer of **\$65 a share**.

- Control premium over the unaffected price = (\$65 − \$50) / \$50 = **+30%** — squarely in the historical 20–40% band.
- A shareholder who bought at the unaffected \$50 makes \$15 a share, a 30% return, realized in cash at close.
- A shareholder who bought *after* the 13D at \$55 makes (\$65 − \$55) / \$55 = **+18%** — still good, but the early reader of the governance captured the extra 12 points.

The takeaway: the control premium is the market price of control, and a takeover defense is precisely the mechanism by which a board can keep that premium from reaching shareholders — so reading the defenses tells you how much of a premium is actually capturable.

![Step-up in share price from an unaffected 50 dollars to 55 dollars after the 13D to a 65 dollar takeout offer with a 30 percent control premium](/imgs/blogs/corporate-governance-fiduciary-duty-and-shareholder-activism-8.png)

## Common misconceptions

Three beliefs about corporate governance are widespread, intuitive, and wrong. Each one, corrected, sharpens the trade.

### "Shareholders run the company"

They do not. As we built in the foundations, shareholders *own* the residual claim but the **board controls** the firm, and management runs it day to day. A retail shareholder with 100 shares has essentially no influence; even a 5% holder has only the *vote* and the *pressure* the 13D campaign generates — not a direct say in operations. This matters for trading because it means *governance structure is decisive*: a "good business" run by an entrenched board for its own benefit can be a bad investment, and a mediocre business with clean governance and an activist on the register can be a great one. The number that makes it concrete: in a dual-class firm, a founder can own **15% of the economics and 60% of the votes** — controlling a company they mostly do not own. Ownership and control are different things, and the gap between them is priced.

### "Activists are short-term raiders who strip companies"

This is the most politically charged misconception, and the evidence is genuinely *mixed* — which is itself the point. The popular image is the 1980s "corporate raider" loading a target with debt and stripping it for parts. But the systematic academic evidence on the modern (post-2000) activist tells a more nuanced story. The large Brav-Jiang-Partnoy-Thomas study and its successors find that, *on average*, targeted firms show **improved operating performance** (return on assets, operating margins) in the years *after* a campaign, and the **~6–7% announcement return does not reverse** over the following year — if activists were merely pumping and dumping, the gains would fade, and on average they do not. That said, the average hides wide variance: some campaigns destroy value, some force buybacks that starve a firm of needed investment, and the debate over whether activism trades long-term health for short-term gains is real and unresolved. The honest summary is *not* "activists are always good" — it is "the lazy 'short-term raider' caricature is contradicted by the average outcome, and you must judge each campaign on its specifics." For trading, that means: do not dismiss an activist on reputation; price the specific plan.

### "Dual-class shares are always bad for shareholders"

The governance-purist view is that dual-class shares are pure entrenchment — a founder keeping control while taking other people's money. And often they are: the discount on poorly-run dual-class firms is real. But the blanket claim is wrong, and the counter-evidence is the list of the most valuable companies on earth. Alphabet, Meta, and Berkshire Hathaway are all dual-class, and a public shareholder who "suffered" the governance discount by buying them anyway earned extraordinary returns, because the founders' long-horizon control let them invest through cycles that short-term-pressured boards would have cut. The empirical literature finds dual-class structures tend to *add* value early in a firm's life (when the founder's vision is the asset) and *destroy* value as the firm matures and the founder's edge fades — which is why **sunset provisions** (clauses that auto-convert super-voting shares to one-vote after, say, 7–15 years or on the founder's departure) are the emerging compromise. The number to hold: a dual-class discount of, say, 10–15% is a *price you pay for the founder's control* — sometimes a bargain, sometimes a trap, and the whole skill is telling which.

## How it shows up in real markets

Strip away the doctrine and four recognizable patterns recur. Each is a place where governance, not the business, moved the price.

### The 13D pop

The cleanest pattern is the announcement jump. When Third Point disclosed its stake in Sotheby's, when Elliott surfaced in any number of names, when Starboard built a position in a sleepy industrial, the stock typically gapped up **5–10% in a day** on nothing but the filing. The 2013 Icahn tweet that he had a "large position" in Apple and that the stock was undervalued moved Apple's market value by tens of billions of dollars *in minutes* — a single sentence from a credible activist, repricing the expected probability that the company would return more cash (which, over the following years, it did, with one of the largest buyback programs in corporate history). The 13D pop is the market pricing the *option* on the campaign before the campaign acts.

### The poison-pill discount and the entrenchment overhang

The mirror image is the stock that *cannot* be unlocked. A firm with a pill, a staggered board, and a controlling family trades cheap to its peers and stays cheap, because the market knows the trapped value has no exit. When such a firm finally *removes* a defense — de-staggers the board, lets a pill expire, or the controlling family sells — the stock often re-rates sharply, because the overhang that suppressed it lifts. The de-staggering wave among S&P 500 firms over the 2010s (driven by index-fund and proxy-adviser pressure) was, company by company, a series of small governance re-ratings.

### The dual-class governance discount

Markets price dual-class entrenchment continuously. Snap's 2017 IPO of **zero-vote** shares was the extreme case — public holders bought economic exposure with *no* governance rights at all — and it triggered S&P Dow Jones Indices and FTSE Russell to **restrict no-vote shares from major indexes**, which itself is a flow event (exclusion from an index removes a buyer base and weighs on the price). Across the dual-class universe, the discount is real but, as the misconceptions section argued, not uniform — the founders who justify it and the founders who abuse it both exist, and the market's job (and yours) is to separate them name by name.

### The control-premium takeout

Finally, the resolution: the takeover. When a company is acquired, the **20–40% control premium** over the unaffected price is the explicit market value of control, paid in cash at close. The Dell fight that opened this post is one version (a founder-led buyout that an activist forced higher); the endless stream of private-equity and strategic buyouts is the broader pattern. For the practitioner, the control premium is both the prize (if you owned the stock at the unaffected price) and the input to the merger-arbitrage trade — once a deal is announced, the question becomes the *probability it closes*, which is the subject of [merger arbitrage and regulatory deal risk](/blog/trading/law-and-geopolitics/merger-arbitrage-trading-regulatory-deal-risk).

A subtle version of the same pattern is the **controlled-company squeeze-out**, where the entity *already* in control — a controlling shareholder or a parent company — tries to buy out the public minority. Because the controller sits on both sides, the duty of loyalty is at its peak and the business-judgment rule's protection evaporates; Delaware applies its most demanding "entire fairness" review (fair price *and* fair process) unless the deal is conditioned up front on both an independent special committee and a *majority-of-the-minority* vote. That legal structure is precisely why a controller cannot simply name a lowball price: the minority's vote and an independent committee are the levers that drag the price toward fair value. When you see a controller propose to take a subsidiary private, the trade is to ask whether those protections are in place — if they are, the offer will likely have to rise; if they are not, expect litigation that does the same job more slowly.

### The Vietnam parallel: state control and the equitization discount

The same control-versus-value arithmetic shows up far from Delaware. In Vietnam, a large share of listed companies are partly **state-owned enterprises** in the middle of **equitization** — the gradual sale of state stakes to the public. A company where the state retains a controlling block carries its own governance discount: minority shareholders cannot force a strategy change, capital allocation answers to policy as much as to returns, and the free float is thin. The *catalyst* that lifts that discount is a **state divestment** — the announcement that the government will sell down its stake, which both adds tradable supply and removes the control overhang. It is the emerging-market cousin of the founder-controlled re-rating, governed by Vietnamese law and the State Securities Commission rather than the Delaware Chancery, but the pricing logic is identical: read who controls the firm and on what terms, and you can size the value the control structure traps. (We take up the Vietnamese machinery — the SSC, foreign-ownership limits, and the equitization program — in the series' [Vietnam stocks track](/blog/trading/vietnam-stocks).)

## How to trade it: the governance playbook

Everything above resolves into a repeatable process: find value trapped by a control structure, estimate the unlock, judge whether the legal machinery makes the unlock feasible, and know what kills the thesis. This is the payoff.

### Step 1 — Screen for the governance discount plus a catalyst

Start with the two ingredients of a value trap: a *discount* and a *catalyst*. Screen for companies trading cheap to their sector or to a sum-of-the-parts estimate — low EV/EBITDA versus pure-play peers, a conglomerate whose segments would re-rate apart, a balance sheet bloated with idle cash. Then look for a **catalyst** that could release it: an activist already on the register (read the 13D filings), a founder near retirement, a sunset clause approaching, a pending de-staggering vote, or pressure from index inclusion rules. A discount with no catalyst is a value trap that stays a trap; a discount *with* a credible catalyst is the setup. Cross-reference the valuation work from the [equity research series](/blog/trading/equity-research) — the governance lens tells you *why* the cheap stock is cheap and what would fix it.

### Step 2 — Value the unlock

Quantify, in dollars, the value that could be released, using the three engines from this post:

- **The cash return**: how much excess cash could be returned, and the EPS accretion from a buyback (the \$2 billion → +25% EPS example above).
- **The breakup**: a sum-of-the-parts model valuing each segment on peer multiples; the gap to the current market cap is the conglomerate discount (the \$100B vs \$75B example).
- **The re-rating**: if clean-governance peers trade at 15x and this firm trades at 10x on the same earnings, the multiple gap times EPS is the re-rating prize (the \$50 → \$75 example).
- **The takeout**: the control premium (20–40% over the unaffected price) if a sale becomes feasible.

Sum the plausible unlocks, weight them by probability, and compare to the current price. If the probability-weighted unlock dwarfs the downside, you have an edge.

### Step 3 — Read the defenses for feasibility

This is the step amateurs skip and professionals live on. *The unlock is only worth its probability of happening,* and the defenses set that probability. Pull the company's charter and bylaws and check:

- **Share structure**: one-share-one-vote, or dual-class? If a founder controls the vote, an activist is nearly powerless — the unlock probability collapses, and the discount may be permanent (unless a sunset or a founder exit is near).
- **Board**: staggered or annually elected? A staggered board means a two-year grind even if the activist is right.
- **Poison pill**: in place or on the shelf? A pill forces negotiation but is redeemable; an activist who wins board seats can pull it.
- **State of incorporation**: Delaware (predictable rules, Unocal/Revlon apply) or somewhere more management-friendly?
- **Recent signals**: a failed say-on-pay vote, a universal-proxy contest, proxy-adviser recommendations against the board — all raise the odds the board can be moved.

A deep discount behind a triple-armored, founder-controlled fortress is *cheap for a reason* and may stay cheap forever. The same discount behind clean governance with an activist already filing is a live trade. The defenses convert the *static discount* into a *probability of unlock*.

### Step 4 — Position and size

How you express the view depends on the catalyst's certainty:

- **Pre-catalyst accumulation**: if you spot the discount before the activist, buy the equity and wait — but size for the possibility the catalyst never comes, because an entrenched board can outlast you.
- **Riding a 13D**: buy after a credible activist files, accepting you have given up the announcement pop in exchange for confirmation that a campaign is live.
- **Event-driven**: once a sale process or proxy fight is underway, the trade narrows toward merger-arb mechanics — the spread becomes a bet on the *probability of completion* (see the [hedge-funds series](/blog/trading/hedge-funds) on event-driven and activist strategies).

Size to the *feasibility-adjusted* unlock, not the headline unlock. A \$25-per-share trapped value with a 30% chance of release is worth roughly \$7.50 of expected value — position to that, not to the \$25.

### What invalidates the thesis

Know your exits before you enter. The governance trade is wrong when:

- **The defenses are stronger than you thought** — a dual-class structure with no sunset, a founder with no intent to leave, a board that out-votes any challenge. The unlock probability is near zero; the discount is structural, not temporary.
- **The activist loses or walks away** — a failed proxy fight, or the activist exits its position (watch for the 13D amendment that discloses a sale). The catalyst is gone; the discount reverts.
- **The "trapped" value was never there** — the sum-of-the-parts assumed peer multiples the segments do not deserve, or the "excess" cash is actually needed for the business. A breakup of bad businesses unlocks nothing.
- **Revlon never triggers** — the board has discretion to *not* sell, and the business-judgment rule protects that choice, so the control premium stays withheld. No sale, no premium.
- **The macro regime turns** — activism and takeovers run on cheap capital and confident credit; when rates spike and financing dries up, buyout premiums shrink and campaigns stall (the link from policy rates to deal flow runs through the same channel as [how the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates)).

The discipline is always the same one this series returns to: read the *rules of control* early, size the value those rules trap or release, and know exactly what would prove you wrong. In governance, the rules are written in the charter, the bylaws, and a century of Delaware case law — and the investor who reads them sees the trapped value before the campaign that frees it.

## Further reading & cross-links

Within this series:

- [Regulatory risk as an asset-pricing factor](/blog/trading/law-and-geopolitics/regulatory-risk-as-an-asset-pricing-factor) — the external-rules cousin of this post: how legal and political risk gets priced as a discount, the same arithmetic applied to antitrust and banned products rather than control.
- [Merger arbitrage: trading regulatory deal risk](/blog/trading/law-and-geopolitics/merger-arbitrage-trading-regulatory-deal-risk) — once a takeover is announced and the control premium is on the table, the trade becomes a bet on the probability the deal closes.
- [Disclosure and accounting law: SOX, IFRS vs GAAP](/blog/trading/law-and-geopolitics/disclosure-and-accounting-law-sox-ifrs-vs-gaap) — the disclosure regime, including the 13D filing, that makes governance legible to outsiders in the first place.

Cross-links out:

- [The equity research playbook](/blog/trading/equity-research) — the valuation toolkit (sum-of-the-parts, multiples, DCF) you use to size a governance unlock.
- [The hedge fund founder's playbook](/blog/trading/hedge-funds) — how event-driven and activist funds actually run these campaigns and size the risk.
- [The 2017 TCJA and the repatriation trade](/blog/trading/law-and-geopolitics/the-2017-tcja-and-the-repatriation-trade) — how a tax law unleashed the record buyback wave that is the lever activists push.
- [How the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates) — the cost-of-capital channel that turns activism and takeovers on and off with the rate cycle.
