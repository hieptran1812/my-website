---
title: "Transfer pricing and offshore profit shifting: how profit moves without the business moving"
date: "2026-08-05"
publishDate: "2026-08-05"
description: "How multinationals price goods, services, loans and intellectual property between their own subsidiaries, why that gives them enormous discretion over where profit appears, and how an outside analyst reads the tax footnote to spot it."
tags: ["forensic-accounting", "transfer-pricing", "tax-avoidance", "effective-tax-rate", "beps", "profit-shifting", "financial-statements", "tax-footnote", "apple", "pillar-two"]
category: "trading"
subcategory: "Forensic Accounting"
author: "Hiep Tran"
featured: true
readTime: 59
---

> [!important]
> **TL;DR** — A multinational is one business but many separate taxpayers. Every time value crosses an internal border it needs a price, and because no outside market set that price, the group has real discretion over which country its profit shows up in. That discretion is usually lawful. It is also, occasionally, where accounting fraud starts.
>
> - The rule is the **arm's-length principle**: charge your own subsidiary what an unrelated party would have charged. The problem is that for unique intellectual property, brand rights and bespoke internal services, there is often no unrelated party to compare to — so the "right" answer is a range, and the taxpayer picks from inside it.
> - The four workhorse channels are **royalties on migrated intellectual property**, **intra-group debt**, **management and service fees**, and **principal structures** that strip local subsidiaries down to a thin, risk-free margin.
> - **Tax avoidance, tax evasion and accounting fraud are three different things.** Most of what follows is the first. Say which one you mean, because the legal consequences are not remotely alike.
> - The analyst's tell is the **geography of profit**: an effective tax rate durably far below the statutory rate, profit per employee that differs by orders of magnitude across jurisdictions, and revenue booked where the group has no customers.
> - The European Commission ordered Ireland in August 2016 to recover **€13 billion** from Apple. The General Court annulled that decision in July 2020. The Court of Justice reinstated it on **10 September 2024**. The same facts produced three different answers from three different benches — which tells you how contested this ground is.
>
> *Law and litigation status stated as of **August 2026**. This area moves: the Double Irish is closed, the UK's diverted profits tax has been folded into corporation tax, and the Pillar Two global minimum tax has acquired a significant carve-out since it took effect. Dates are given throughout for that reason.*

Here is a question that sounds naive and is not.

A phone is designed in California, made by a contractor in China, shipped to a warehouse in the Netherlands, and sold to a customer in Munich for €1,000. Somewhere between the design and the sale, roughly €300 of profit is created. Which country gets to tax it?

There is no natural answer. The profit does not have a physical location the way the warehouse does. It is the residue left over after you subtract costs from revenue, and both of those are spread across a dozen countries. So the tax system does the only thing it can: it makes each legal entity in the group file its own return, in its own country, and it requires the group to put a **price** on every transaction between its own entities. The German sales company buys the phone from somebody. The designer licenses the design to somebody. Each of those internal prices moves profit from one tax jurisdiction to another, one euro at a time.

That set of internal prices is called **transfer pricing**, and it is the largest single lever a multinational has over its own tax bill.

![The group structure and its two flows: real money in from customers at the bottom, paper flows moving profit to the top](/imgs/blogs/transfer-pricing-and-offshore-profit-shifting-1.webp)

The diagram above is the mental model for the whole post. Look at it as two circulatory systems laid on top of each other. The first is real: customers hand over money at the bottom, and that money buys goods that a factory somewhere actually made. The second is paper: royalties, service charges, interest and transfer prices flow *upward* to entities that may have a dozen employees and no customers at all. The first system is the business. The second system decides who gets taxed on it.

Nothing on that diagram is inherently improper. Groups genuinely do centralise their intellectual property, genuinely do run regional trading hubs, and genuinely do charge each other for shared services. The forensic question is never "is there a royalty?" It is: **is the royalty the size an unrelated party would have agreed to, and is anybody actually doing anything in the box that receives it?**

## The building blocks: why a price exists inside a company at all

Before any of the tricks make sense, you need five ideas. If you already know them, skim; if you do not, none of the rest will land.

### One business, many taxpayers

When you look at a listed multinational, you see one set of consolidated accounts: one revenue line, one profit line, one tax line. Consolidation deliberately erases everything the group did with itself. If the Irish subsidiary sells something to the German subsidiary for \$94, that sale and that purchase cancel out. From the outside, they never happened.

Tax law does not consolidate. It sees the legal entities — the Irish company, the German company, the Swiss company — and taxes each one separately, in its own country, on its own profit. So a transaction that vanishes from the consolidated income statement is, for tax purposes, extremely real. It moved taxable profit across a border.

That gap between how accounting sees a group (one entity) and how tax sees it (dozens of entities) is the space the whole industry lives in.

Two terms you will need immediately:

- An **intra-group** or **related-party** transaction is any transaction between entities under common control — parent to subsidiary, subsidiary to subsidiary, or with an entity a controlling shareholder also controls. The broader forensic treatment of these is in [related-party transactions and self-dealing](/blog/trading/forensic-accounting/related-party-transactions-and-self-dealing); here we care specifically about the cross-border, tax-motivated kind.
- A **transfer price** is the price put on such a transaction. It applies to goods, services, loans, guarantees, brand rights, patents, software, and the use of a customer list. Anything of value that crosses an internal border needs one.

### The arm's-length principle

The governing rule, in almost every country on earth, is the **arm's-length principle**: price a transaction with your own subsidiary the way you would have priced it with a stranger.

It appears in Article 9 of the OECD Model Tax Convention, which lets a tax authority adjust profits where "conditions are made or imposed between the two enterprises in their commercial or financial relations which differ from those which would be made between independent enterprises." In the United States it is section 482 of the Internal Revenue Code, which lets the IRS reallocate income among commonly controlled entities to prevent evasion of taxes or to clearly reflect income. The detailed how-to is the *OECD Transfer Pricing Guidelines for Multinational Enterprises and Tax Administrations*, most recently consolidated in the 2022 edition, which most countries either adopt directly or mirror in domestic rules.

The principle is elegant and, in a large fraction of real cases, unusable. It asks: what would an independent party have charged? For a tonne of copper concentrate or a barrel of Brent, there is a published market price and the question is nearly answerable. For "the right to use the iPhone brand in France", or "the software architecture that a company spent fifteen years building and has never licensed to anyone", there is no independent party, because the thing has never been sold to one. The comparison the law demands does not exist.

So the practical outcome is not a single correct price. It is a **defensible range**, and the group picks a point in it. That is the crucial thing to internalise: intra-group prices are inherently discretionary, not because the law is weak, but because the underlying question — what is a unique thing worth to a stranger who has never been offered it? — has no unique answer.

### The tax vocabulary

Five terms, each defined once:

- **Statutory rate** — the headline corporate income tax rate a country charges. Ireland 12.5% on trading income; the United States 21% federal since the 2017 tax reform, down from 35%; Germany roughly 30% once the trade tax is included; Bermuda, historically, zero.
- **Effective tax rate (ETR)** — total income tax expense divided by pre-tax profit, as reported in the accounts. This is the number that actually reduces earnings per share. A group operating in high-tax countries with a durably low ETR is telling you something.
- **Current versus deferred tax** — *current* tax is what is payable on this year's tax returns; *deferred* tax is an accounting accrual for timing differences between the books and the tax return. Total tax expense is the sum. A company can have a high book ETR and pay very little cash, or vice versa, and the difference matters.
- **Withholding tax** — a tax a country levies at the border when a payment (a dividend, interest, or a royalty) leaves it. Tax treaties and, inside the European Union, the Interest and Royalties Directive reduce or eliminate it between certain pairs of countries. Withholding tax is the friction that a lot of structuring is designed to route around.
- **Permanent establishment (PE)** — the threshold of physical or contractual presence at which a foreign company becomes taxable in a country. Staying below it — selling *into* a country rather than *in* it — is a classic structuring goal.

One more: a **controlled foreign company (CFC) rule** is a domestic rule that taxes the parent on certain passive or mobile income earned by its foreign subsidiaries, whether or not that income is repatriated. CFC rules are the main defence a home country has against profit parked abroad, and every structure in this post is partly an exercise in staying outside them.

### Three different things: avoidance, evasion, and accounting fraud

This is the most important paragraph in the article, and it is the one most commentary gets wrong.

**Tax avoidance** is arranging your affairs, within the law, to pay less tax. Choosing to hold your patents in Ireland rather than California is avoidance. It is legal. It may be aggressive, it may be unpopular, it may be reversed years later by a court, but a company doing it is not committing a crime.

**Tax evasion** is illegal: concealing income, falsifying documents, lying to a tax authority. Different conduct, criminal consequences.

**Accounting fraud** is a third thing again: misstating the financial statements that investors rely on. A company can avoid tax perfectly legally and report it perfectly honestly. It can also avoid tax legally and then *misreport* the resulting position — by under-reserving for a tax dispute it expects to lose, by releasing tax reserves into earnings to hit a quarter, or by describing a structure in the footnotes in a way that hides the risk. That is where transfer pricing crosses into this series' territory.

There is also a fourth, rarer case: where the intra-group pricing is not tax planning at all but a vehicle for fabricating revenue — booking sales to a "customer" that is really a controlled entity. That is a different fraud, covered in [round-tripping and fabricated revenue](/blog/trading/forensic-accounting/round-tripping-and-fabricated-revenue) and in [shell companies, reverse mergers and how fraud gets listed](/blog/trading/forensic-accounting/shell-companies-reverse-mergers-and-how-fraud-gets-listed).

When you write about a company's tax structure, say which of the four you are alleging. "Apple avoided tax" and "Apple committed fraud" are not variations on a theme; the second is defamatory if you cannot prove it, and in the Apple case nobody has ever alleged it.

#### Worked example: one \$100 sale, three invoices

Everything above is abstract until you follow a single unit of product through a group. Here is that walkthrough. The numbers are illustrative arithmetic, not a claim about any real company.

![The invoice chain: one \$100 sale, three intra-group invoices, and where the margin lands](/imgs/blogs/transfer-pricing-and-offshore-profit-shifting-4.webp)

A group makes a widget and sells it in Germany for \$100. Three of its entities touch it:

1. **The factory**, in Country M, where the corporate tax rate is 20%. It costs \$60 to make the widget. The factory sells it to a group trading hub for \$66. Its profit is \$6, and its tax is \$6 × 20% = **\$1.20**.
2. **The trading hub**, in Switzerland, which has negotiated an effective rate of about 5% under a tax ruling. (Switzerland abolished its special cantonal company regimes in the 2019 reform that took effect in 2020, so treat the 5% as an illustrative historical rate, not a current one.) It buys at \$66, incurs \$3 of its own costs, and sells to the German distributor at \$94. Its profit is \$94 − \$66 − \$3 = \$25, and its tax is \$25 × 5% = **\$1.25**.
3. **The distributor**, in Germany, where the rate is 30%. It buys at \$94, spends \$3 on selling costs, and sells to the customer for \$100. Its profit is \$100 − \$94 − \$3 = \$3, and its tax is \$3 × 30% = **\$0.90**.

Group totals: revenue \$100, total costs \$66 (\$60 + \$3 + \$3), pre-tax profit \$34. Total tax \$1.20 + \$1.25 + \$0.90 = **\$3.35**. The group's effective tax rate is \$3.35 ÷ \$34 = **9.9%**.

Now change exactly one thing: assume the Swiss hub is what it looks like — a small team routing paperwork, taking no inventory risk and owning no intellectual property. Under the arm's-length principle such an entity should earn a routine service return, say cost plus 10% on its \$3 of costs, which is \$0.30 of profit. The residual \$24.70 belongs wherever the risk and the intangibles actually sit; assume that is the factory in Country M.

- Factory: profit \$6 + \$24.70 = \$30.70, tax at 20% = **\$6.14**
- Hub: profit \$0.30, tax at 5% = **\$0.015**
- Distributor: unchanged, tax **\$0.90**

Total tax = **\$7.06**, and the effective rate is \$7.06 ÷ \$34 = **20.8%**.

The widget is identical. The factory, the customer, the shipping route and the \$100 price are identical. Group profit is \$34 either way. The only thing that changed is a number on an internal invoice, and it doubled the group's tax bill.

*The intuition: transfer pricing does not create or destroy profit. It relocates it, and relocation is worth exactly the difference between two tax rates.*

## The five methods, and where the judgement lives

The OECD sanctions five methods for testing whether an intra-group price is arm's length. Knowing them matters because a company's choice of method tells you a great deal about how much room it has given itself. They are worth reading in order, because the order is the point: each step down the list replaces an observable market fact with a judgement somebody made.

**1. Comparable uncontrolled price (CUP).** Find the price at which the same or a very similar thing changed hands between unrelated parties, and use it. This is the gold standard and the least common in practice, because "the same thing" is a high bar. It works for commodities, standard components, and simple loans where a published benchmark rate exists. It does not work for a bespoke drug compound.

**2. Resale price method.** Start with the price at which the related-party buyer resells to an outside customer, and work backwards by subtracting an arm's-length gross margin. Used for distributors that add little value. The discretion sits in which independent distributors you accept as comparable.

**3. Cost plus.** Take the supplier's costs and add a markup that independent suppliers earn on similar work. Used for contract manufacturers and service providers. The discretion sits in *which costs go into the base* — a markup of 8% on a cost base that quietly excludes half the real costs is not an 8% markup.

**4. Transactional net margin method (TNMM).** Compare the *net* profit margin the tested entity earns (operating profit over sales, or over costs, or over assets) with the range of net margins earned by a set of independent companies doing broadly similar things. This is the workhorse: probably the majority of the world's transfer-pricing documentation rests on it. It is also the softest, because you get to choose the comparable set, the profit-level indicator, and a battery of "comparability adjustments".

**5. Profit split.** Where both sides contribute unique value — two research centres that jointly built a product, say — you compute the combined profit and split it according to each side's relative contribution. The method is conceptually the most honest for genuinely integrated businesses. It is also the one where the answer is most obviously a negotiation, because the splitting factors (headcount? research spend? capitalised development cost?) are chosen, not observed.

![The five OECD transfer-pricing methods, from most objective to most discretionary](/imgs/blogs/transfer-pricing-and-offshore-profit-shifting-3.webp)

Laid out side by side, the ladder is easier to hold: objectivity falls as you move down it, and discretion rises to meet you. A group that tests its most valuable transactions with CUP is standing on market evidence. A group that tests them with TNMM or a profit split has chosen the rungs where the answer is argued rather than observed — which is not wrongdoing, but is where you should look first.

### The benchmarking study and the interquartile range

In practice, a TNMM analysis works like this. A consultant screens a commercial database of company accounts for independent companies in similar industries and functions, throws out the ones that are loss-making or too large or too small, and ends up with perhaps 15 to 30 comparables. Their operating margins are ranked, and the **interquartile range** — the middle 50%, from the 25th to the 75th percentile — is treated as the arm's-length range.

If the tested entity's margin falls anywhere inside that range, the price is defensible. If it falls outside, the company adjusts to the median.

Sit with what that means. Suppose the range for a limited-risk distributor is 1.2% to 3.8% operating margin. On €1,000m of sales, a company can lawfully report anywhere between €12m and €38m of local profit and be inside the range either way. A €26m swing in German taxable profit, on identical trading, produced by nothing more than where in the range you choose to sit. Repeat that across thirty countries and the aggregate discretion is enormous — and every single euro of it is inside the rules.

That is why "the company's transfer pricing was found to be within the arm's-length range" is a much weaker statement than it sounds. It usually is.

### Advance pricing agreements and tax rulings

Because the range is wide and the consequences are large, companies often ask a tax authority to bless the arrangement in advance. A bilateral **advance pricing agreement (APA)** binds two tax authorities and the taxpayer to a method for several years. A unilateral **tax ruling** binds one.

APAs are a legitimate certainty mechanism used by thousands of groups, and they are slow. HMRC's statistics for the 2024–25 tax year, published on 11 March 2026, record **26 advance pricing agreements** reached, at an average of **43.9 months** each. Nearly four years to agree a method with one tax authority is the clearest available measure of how genuinely contested the arm's-length range is — nobody spends four years negotiating a number that was obvious. The same release puts HMRC's transfer-pricing yield for the year at **£3,387 million**.

But rulings are also the artefact that made the European state-aid cases possible: when a member state gives one company a ruling that lets it compute profit in a way no other company gets, the Commission's argument is that the ruling is a **selective advantage** — a subsidy dressed as a tax opinion. Whether that argument works has turned out to depend, as we will see, on very fine questions about what the national law actually said.

## Channel one: migrating the intellectual property

If you had to point to one structural move that explains most of the profit that ends up in low-tax jurisdictions, it is the migration of intellectual property.

The logic is simple. In a modern business — pharmaceuticals, software, consumer electronics, branded consumer goods — most of the value is not in the factory. It is in the patents, the brand, the algorithms, the customer relationships. Those assets are **mobile**: a patent can be owned by a company registered anywhere. A steel mill cannot.

So a group creates an entity in a low-tax jurisdiction and arranges for that entity to own, or co-own, the group's intangibles. Every operating company that uses those intangibles then pays it a royalty. The operating companies keep a modest, defensible return for their "routine" functions — selling, distributing, manufacturing to specification — and the residual profit, which is most of it, accumulates in the holding entity.

The two ways to get the IP there:

**Sell it.** The parent sells the intangibles to the low-tax entity for a lump sum. The problem is that the parent then has a large taxable gain in a high-tax country, valued at what the IP is worth *today*, which for a successful product is a great deal.

**Cost-sharing.** Far more common. The parent and the offshore entity enter a **cost-sharing arrangement** (in US terms, a cost-sharing arrangement or CSA; the OECD calls them cost contribution arrangements). They agree to jointly fund future research and development and to split the resulting rights by territory: the parent takes the home market, the offshore entity takes the rest of the world. The offshore entity pays a **buy-in** or **platform contribution** payment for the existing IP it is getting access to, plus its share of ongoing R&D.

Cost-sharing is powerful because the buy-in is priced on the value of what exists *at the time of the transfer*. Migrate the IP early — while a drug is in phase II, while a product line is still small — and the buy-in is modest. If the product then becomes enormous, all that upside accrues offshore, at the offshore rate, and the tax authority's chance to tax it has passed.

This is exactly why the OECD's 2015 BEPS work (Actions 8 to 10) reframed intangibles around **DEMPE functions** — development, enhancement, maintenance, protection and exploitation. The new question is not "who holds legal title to the patent?" but "who actually performs and controls the functions that make the patent valuable, and who bears the financial risk?" An entity that holds title but has no people cannot, under the current guidelines, be entitled to the residual return. Legal ownership without substance became much harder to defend after 2015. Note the word *harder*, not *impossible*: a "substance" requirement is satisfied by putting real decision-makers in the jurisdiction, which several groups promptly did.

#### Worked example: the royalty strip

Three European operating companies — Germany, France and the UK — sell the group's products. Combined, they book €1,000m of revenue and €700m of operating costs, giving €300m of pre-royalty operating profit. Their blended local tax rate is 28%.

The group's IP is held by a holding company that is incorporated in Ireland but managed from Bermuda, so it is not tax resident in Ireland and pays no tax. The operating companies pay it a royalty of **25% of revenue** for the right to use the brand and the technology.

**With the royalty:**

- Royalty paid = €1,000m × 25% = **€250m**
- Local profit after royalty = €300m − €250m = **€50m**
- Local tax = €50m × 28% = **€14m**
- IP holding company: receives €250m, has negligible costs, pays **€0**
- Group pre-tax profit €300m, group tax €14m, **effective rate 4.7%**

**Without the royalty**, the €300m would have been taxed locally at 28% = **€84m**.

So the royalty is worth €70m a year. Now the forensic question: is 25% of revenue an arm's-length royalty?

That depends entirely on the industry. Third-party licences for brand and technology in consumer and industrial sectors commonly run in the low single digits to high single digits as a percentage of net sales — the figure varies enormously by sector, and you would need the actual comparable licences to say. Suppose the defensible rate here were **6%**:

- Royalty = €1,000m × 6% = **€60m**
- Local profit = €300m − €60m = **€240m**, taxed at 28% = **€67.2m**
- Group effective rate = €67.2m ÷ €300m = **22.4%**

The difference between a 6% royalty and a 25% royalty is €53.2m of tax a year, on exactly the same trading. Neither rate is obviously fraudulent. One is obviously more aggressive.

*The intuition: a royalty rate is a dial that sets how much profit stays in the country where the customers are. Nothing about the dial is visible in the consolidated accounts, because the royalty is eliminated on consolidation.*

There is a legitimate variant worth naming, because it muddies the picture: the **patent box**. Several countries deliberately offer a reduced rate on income from qualifying intellectual property in order to attract R&D. Ireland's Knowledge Development Box, introduced in Finance Act 2015, taxed qualifying IP income at an effective 6.25% — half the 12.5% trading rate — when it was brought in. That rate has since been revised upward as the global minimum tax made sub-15% regimes pointless for large groups, so verify the current figure before you quote it; what matters here is the design, not the number. The UK, the Netherlands, and others have their own equivalents. Under the OECD's "modified nexus approach", the benefit is now supposed to be proportionate to the R&D actually done in the country. A patent box is a policy choice by a parliament, not a loophole — which is a reminder that "low tax rate on IP income" is not by itself evidence of anything improper.

## Channel two: intra-group debt and thin capitalisation

The second channel exploits an asymmetry that sits at the heart of every corporate tax system in the world: **interest is deductible; dividends are not.**

If a parent funds its German subsidiary with €1,000m of equity, the subsidiary earns profit and pays 30% tax on all of it; whatever is left can be paid up as a dividend, out of after-tax money. If the parent instead funds the subsidiary with €200m of equity and an €800m loan, the subsidiary pays interest to the group before tax, deducts it, and taxes only what is left. The interest arrives at the lender — often a group finance company in Luxembourg, Ireland, Switzerland or Hungary — where it may be taxed lightly or, historically, barely at all.

The business has not changed. The plant, the staff, the customers and the operating profit are all identical. Only the capital structure moved, and the capital structure is entirely within the group's gift. Loading a subsidiary with intra-group debt like this is called **thin capitalisation**.

![The same subsidiary, funded with equity versus funded with intra-group debt](/imgs/blogs/transfer-pricing-and-offshore-profit-shifting-7.webp)

#### Worked example: the same subsidiary, funded two ways

A subsidiary in a 30%-tax country has \$1,000m of assets and generates \$150m of EBITDA (earnings before interest, tax, depreciation and amortisation). Assume no depreciation, so EBITDA and EBIT are both \$150m.

**Case A — funded with \$1,000m of equity:**

- EBIT \$150m, interest \$0, taxable profit \$150m
- Tax at 30% = **\$45m**
- Group tax bill: **\$45m**

**Case B — funded with \$200m of equity and an \$800m intra-group loan at 8%:**

- Interest = \$800m × 8% = **\$64m**
- Taxable profit = \$150m − \$64m = **\$86m**
- Local tax at 30% = **\$25.8m**
- The group finance company receives \$64m and is taxed at an effective 1% = **\$0.64m**
- Group tax bill: \$25.8m + \$0.64m = **\$26.44m**

The saving is \$45m − \$26.44m = **\$18.56m a year**, from a financing decision that changed nothing about the business.

This is precisely why every major jurisdiction now caps the deduction. The dominant design, recommended by BEPS Action 4 and enacted in the EU's Anti-Tax Avoidance Directive (Directive (EU) 2016/1164, known as ATAD) and in section 163(j) of the US Internal Revenue Code, is a **fixed-ratio rule**: net interest deductions are limited to roughly 30% of EBITDA, with a small-company safe harbour (€3m under ATAD) and a carry-forward for the disallowed amount.

**Case B under a 30%-of-EBITDA cap:**

- Cap = \$150m × 30% = **\$45m** deductible
- Disallowed = \$64m − \$45m = **\$19m**, carried forward
- Taxable profit = \$150m − \$45m = \$105m, tax at 30% = **\$31.5m**
- Plus the finance company's \$0.64m: group bill **\$32.14m**

The cap recovers most, but not all, of the leakage. And note the second lever the rule does *not* touch: the **interest rate itself**. An 8% rate on an intra-group loan to a subsidiary that could have borrowed from a bank at 4% moves twice as much profit as the arm's-length rate would. Pricing the rate is a transfer-pricing question, and it is exactly the question the Australian courts answered in the Chevron case discussed later.

*The intuition: the tax system rewards debt over equity, and a group gets to decide how much of its own debt to point at its own high-tax subsidiaries.*

## Channel three: management fees, service charges, and the cost base

The third channel is the least glamorous and, in a lot of mid-sized groups, the most used, because it needs no intellectual property and no financing structure. It is a bill.

Headquarters provides services to subsidiaries: strategy, IT, procurement, HR, legal, treasury, brand marketing. It charges them out. Those charges are deductible in the paying country and taxable in the receiving one, and if the receiving one is a low-tax jurisdiction, the arithmetic is familiar.

The arm's-length constraints here are meant to be real. The OECD guidelines require that:

1. A service was actually **rendered** — the "benefit test". A charge for the shareholder's own activities (preparing consolidated accounts, investor relations, the cost of the parent's stock exchange listing) is a **shareholder cost** and is not chargeable to subsidiaries at all.
2. The charge is **allocated** on a sensible key (revenue, headcount, transactions) rather than by whatever balances the model.
3. The markup on cost is modest for routine services. The OECD's simplified approach for low-value-adding intra-group services suggests a **5% markup** with no benchmarking study required.

What actually happens in aggressive cases: the fee is a fixed percentage of the subsidiary's revenue rather than a share of real cost; the same activity is charged twice, once as a management fee and once inside a royalty; the cost base is padded with shareholder costs; and the documentation describing what was delivered is a single page with no time records behind it.

The reason to know this channel: it is the one that shows up in smaller and emerging-market groups where there is no patent to migrate. If a listed subsidiary pays 4% of revenue to a privately held parent entity for "management services", you are looking at value leaving the listed vehicle and arriving at the controlling shareholder. That is a governance problem before it is a tax problem, and it is dealt with at length in [related-party transactions and self-dealing](/blog/trading/forensic-accounting/related-party-transactions-and-self-dealing).

## Channel four: the principal structure and the limited-risk distributor

The fourth channel is the most systematic, and it is what the previous three add up to when a group redesigns itself deliberately.

In a **principal structure**, the group designates one entity — typically in a low-tax or ruling-friendly jurisdiction — as the "principal" or "entrepreneur". The principal is contractually deemed to own the inventory, bear the market risk, own or license the intangibles, and direct the strategy. Every other entity is converted, by contract, into a service provider to the principal:

- The former manufacturer becomes a **toll** or **contract manufacturer**, paid cost plus a few percent.
- The former distributor becomes a **limited-risk distributor** (LRD) or a **commissionaire**, earning a thin fixed margin on sales.
- The former R&D centre becomes a **contract researcher**, paid cost plus a markup, with the resulting IP owned by the principal.

Under the arm's-length principle this is defensible on its own terms, because return should follow risk: if the principal really does bear the inventory and market risk, it deserves the residual. The catch is that risk allocation is done by contract, and contracts between related parties are written by the same people on both sides. The BEPS Actions 8–10 revisions attacked exactly this by requiring that an entity can only be allocated a risk if it has the people who **control** that risk and the **financial capacity** to bear it — the so-called "cash box" rules aimed at entities with capital and no competence.

For an analyst, the tell is a step change in a subsidiary's reported margin with no change in its business. If a group's German company earned 8% operating margins for a decade and then, after a "European business restructuring", earns 2.5% on the same sales with the same staff and the same customers, a principal structure was almost certainly implemented. That restructuring itself can be a taxable event — several European tax authorities now assess an "exit charge" for the transfer of profit potential — and it is the kind of thing that appears in the MD&A as a one-line reference to "supply chain optimisation". Reading past that phrasing is the subject of [the footnotes and MD&A: where the bodies are buried](/blog/trading/forensic-accounting/the-footnotes-and-mda-where-the-bodies-are-buried).

## The Double Irish with a Dutch Sandwich

The most famous structure in international tax deserves its own section, partly because it is a beautiful illustration of how the channels combine and partly because it is *gone* — which makes it safe to describe in detail.

![The Double Irish with a Dutch Sandwich: two Irish companies, one Dutch conduit, and a Bermuda destination](/imgs/blogs/transfer-pricing-and-offshore-profit-shifting-5.webp)

The structure had four parts. Everything in this section is written about the past: the Double Irish was closed to new entrants in 2015 and the grandfathering ran out on 31 December 2020. It does not operate today, and a group described as "using the Double Irish" in the present tense is being described inaccurately.

**1. Two Irish companies, one of which was not Irish for tax.** Until Ireland changed its rules, a company incorporated in Ireland was tax resident where it was **managed and controlled**, not where it was registered. So a group could register "Ireland Holdings" in Dublin, hold its board meetings in Bermuda, and have a company that was Irish enough to sit inside EU structures but was tax resident in a jurisdiction with no corporate income tax. That is the first "Irish". The second is a genuinely Irish-resident operating company that employs people, signs customer contracts, and pays Irish tax at 12.5%.

**2. The IP sat in the non-resident one.** Ireland Holdings owned the non-US rights to the group's intangibles, usually acquired through a cost-sharing arrangement with the US parent. It licensed them onward.

**3. The operating company kept almost nothing.** Ireland Operating booked the group's non-US sales. It paid Ireland Holdings a royalty large enough to strip its profit down to a routine cost-plus return on its own operating costs. It paid Irish tax on that thin sliver.

**4. The Dutch sandwich existed to break a withholding tax.** If Ireland Operating paid the royalty directly to a Bermuda-resident company, Ireland would have levied withholding tax on the way out. So the royalty went first to a Dutch BV — a company with, typically, no employees. Ireland to the Netherlands is an EU-internal payment, exempt from withholding under the Interest and Royalties Directive; and the Netherlands, at the time, levied no withholding tax on outbound royalties at all. The Dutch company kept a tiny spread and passed the rest to Bermuda. Hence the sandwich: two Irish slices, Dutch filling.

Underneath all of this sat a US rule known as **check-the-box**, which let the group elect to have the offshore subsidiaries disregarded for US tax purposes, so that the royalty payments between them were invisible to the US anti-deferral rules that would otherwise have taxed them.

#### Worked example: \$10bn of sales, \$16.25m of tax

Illustrative arithmetic, built on the structure above.

The group books \$10,000m of non-US revenue through Ireland Operating, which incurs \$1,500m of its own operating costs and is entitled, under its transfer-pricing policy, to a cost-plus-8% return.

- Ireland Operating's profit = \$1,500m × 8% = **\$120m**; Irish tax at 12.5% = **\$15m**
- The royalty is whatever is left: \$10,000m − \$1,500m − \$120m = **\$8,380m**, paid to the Dutch BV
- The Dutch BV keeps a **\$5m** spread and passes **\$8,375m** onward; Dutch tax at 25% on \$5m = **\$1.25m**
- Ireland Holdings, resident in Bermuda, receives \$8,375m and pays **\$0**

Group pre-tax profit on this activity = \$10,000m − \$1,500m = **\$8,500m**. Total tax paid = \$15m + \$1.25m = **\$16.25m**. Effective rate = **0.19%**.

*The intuition: each step in the chain is individually unremarkable — a royalty, a conduit, a residence rule — and the combination produces a rate two orders of magnitude below any statutory rate in the chain.*

### Why it died

Three separate hits, in order:

- **Ireland closed the residence loophole.** Following the October 2014 Budget, Ireland legislated that companies incorporated in Ireland are Irish tax resident. The change applied to newly incorporated companies from 1 January 2015, with existing structures grandfathered until **31 December 2020**.
- **The Netherlands closed the conduit.** From 1 January 2021 the Netherlands introduced a conditional withholding tax on interest and royalty payments to low-tax and blacklisted jurisdictions, at a rate matching its top corporate rate. The Dutch filling stopped being free.
- **The United States changed the incentive.** The 2017 US tax reform cut the federal rate from 35% to 21%, imposed a one-time transition tax on accumulated offshore earnings, and created **GILTI** (global intangible low-taxed income), a minimum tax on the offshore intangible income of US groups. Parking IP offshore stopped being nearly as valuable to a US parent.

Reuters, reporting from Dutch corporate filings, said Google's Dutch conduit company routed **€19.9 billion** to Bermuda in 2017 and **€21.8 billion** in 2018. Google announced at the end of 2019 that it would dismantle the licensing structure. The structure worked exactly as designed, for about a decade and a half, and then the three governments whose rules it depended on each changed one rule.

## Where tax planning becomes accounting fraud

Everything so far has been avoidance: aggressive, contested, expensive to defend, but not fraud. This series is about fraud, so it is worth being precise about the crossings.

**The tax position is misstated.** Under US GAAP, a company may only recognise the benefit of an uncertain tax position if it is **more likely than not** to be sustained on examination, and then only at the largest amount with a greater-than-50% likelihood of being realised (ASC 740-10, the standard originally issued as FIN 48). IFRS reaches a similar place through IFRIC 23. A company that knows its Swiss structure is under audit and reserves nothing against it is not avoiding tax; it is overstating earnings. This is the single most common way transfer pricing becomes an accounting problem.

**The reserve is used as a cookie jar.** Tax reserves are estimates, and estimates are releasable. Releasing a tax reserve because a statute of limitations expired is proper and common. Releasing one because the quarter is two cents short is earnings management, and it is invisible unless you track the reserve balance year over year. See [cookie-jar reserves and big-bath accounting](/blog/trading/forensic-accounting/cookie-jar-reserves-and-big-bath-accounting) for the general pattern.

**The disclosure hides the exposure.** The tax footnote is required to describe material uncertain positions and the range of reasonably possible change over the next twelve months. A footnote that says "the Company is subject to examination in various jurisdictions" and nothing else, while the company carries a nine-figure exposure it expects to lose, is a disclosure failure.

**The structure is used to move value, not just tax it.** Where an offshore entity is not merely holding IP but is buying goods from the listed company at inflated prices, or selling to it at inflated prices, and that entity is controlled by an insider rather than by the group, the transfer price is a siphon. This is the classic emerging-market pattern, and it is why an auditor's first question about a related-party entity is *who owns it*, not *what did it pay*. Related structures and how they get concealed are covered in [off-balance-sheet financing and special purpose entities](/blog/trading/forensic-accounting/off-balance-sheet-financing-and-special-purpose-entities).

Legally, there is also a middle ground that is neither ordinary avoidance nor fraud: doctrines that let a court disregard the form of a transaction. The US **economic substance doctrine**, codified in section 7701(o) of the Internal Revenue Code in 2010, denies tax benefits to transactions that lack a substantial non-tax purpose and do not meaningfully change the taxpayer's economic position. The EU's ATAD requires member states to have a **general anti-abuse rule**. The UK's **diverted profits tax (DPT)** applied to profits arising from 1 April 2015 and was charged at 31% from 1 April 2023 — a deliberate six-point premium over the 25% main corporation tax rate — precisely so that contrived arrangements cost more than paying ordinary corporation tax. Note the tense. HMRC's transfer pricing and DPT statistics published on 11 March 2026 record that **DPT is repealed and replaced by the Unassessed Transfer Pricing Profits (UTPP) rules for accounting periods beginning on or after 1 January 2026**, and that the new rules "retain the essential features of DPT but form part of the CT regime". The premium survives; the separate tax does not. The same statistics put DPT's cumulative haul at **over £10.5 billion** from 2015–16 to 2024–25, against only **£94 million** in 2024–25 itself — the pattern of a deterrent that worked by changing behaviour rather than by collecting.

None of these doctrines make the taxpayer a criminal. They make the structure not work.

## The analyst's angle: the geography of profit

You will almost never see a company's transfer-pricing documentation. It is confidential, voluminous, and filed with tax authorities, not investors. What you can see is the residue it leaves in the financial statements, and the residue is surprisingly informative.

The core idea: **profit should be geographically near the things that produce it** — the customers, the employees, the factories, the research. When it is not, something is routing it, and the routing is either a legitimate structure you should understand or a risk you should price.

### Reading the effective-versus-statutory tax rate reconciliation

Every set of audited accounts contains a table reconciling the tax you would expect at the statutory rate to the tax actually recorded. Under US GAAP it is required whenever items are individually significant; under IAS 12 it is required outright. It is the single most useful disclosure in the entire filing for this purpose, and most readers skip it.

![The tax-rate reconciliation read as a bridge from statutory 21% to effective 12.4%](/imgs/blogs/transfer-pricing-and-offshore-profit-shifting-6.webp)

#### Worked example: reading a 12.4% effective rate

A US-parented group reports \$10,000m of pre-tax income and \$1,240m of total income tax expense — an effective rate of 12.4% against a 21% federal statutory rate. Its reconciliation looks like this (illustrative):

| Reconciling item | Percentage points | \$m |
| --- | ---: | ---: |
| US federal statutory rate | 21.0 | 2,100 |
| Foreign rate differential | (6.8) | (680) |
| Foreign-derived intangible income (FDII) deduction | (1.5) | (150) |
| Excess tax benefit on stock compensation | (1.2) | (120) |
| Uncertain tax positions | 0.9 | 90 |
| State and local taxes, net | 0.6 | 60 |
| Other, net | (0.6) | (60) |
| **Effective tax rate** | **12.4** | **1,240** |

Read it line by line:

- **The foreign rate differential is the structure, priced.** It says: our foreign profit is taxed at rates far below 21%, and that is worth **\$680m a year**. This is the number to write down. If the structure unwinds — a court ruling, a law change, a Pillar Two top-up — up to \$680m of annual after-tax earnings is at risk. On a 20× multiple that is a meaningful fraction of the market capitalisation, and almost nobody models it.
- **FDII is a US policy incentive**, not offshore planning: it is a deduction for income from serving foreign markets from US-held intangibles. It is a benefit granted *for keeping IP onshore*, and it is durable in a way the foreign rate differential is not.
- **The stock-compensation benefit is not a structure at all.** It arises when the share price rises between grant and vesting, and it reverses when the share price falls. Treating it as recurring is a common modelling error.
- **The uncertain-tax-position line is a cost, and it is a signal.** A company adding 0.9 points of tax expense for positions it may lose is telling you the structure is contested.

*The intuition: the reconciliation is the only place a company quantifies, in one number, what its offshore structure is worth per year — and therefore how much is at stake if it stops working.*

The follow-up test is the five-year one. A single low year is noise: a legal settlement, a one-off release, a valuation-allowance change. A foreign rate differential that sits between −5 and −8 points for five consecutive years is a permanent architectural feature, and it should be visible in the group's legal-entity structure and its country-by-country data.

### Cash taxes versus book taxes

The reconciliation explains the *book* rate. Separately, compute the **cash tax rate**: cash taxes paid (disclosed in the cash flow statement or its supplemental disclosures) divided by pre-tax income.

The two diverge for legitimate reasons — accelerated depreciation, loss carry-forwards, timing of estimated payments. But a company whose cash rate runs persistently far below its book rate is deferring tax somewhere, and a company whose cash rate runs persistently *above* its book rate may be paying assessments it has not yet recognised in earnings. Either direction is worth a footnote read. The general principle — that cash is harder to fake than accrual — is the argument of [reading the cash flow statement: why cash beats net income](/blog/trading/forensic-accounting/reading-the-cash-flow-statement-why-cash-beats-net-income).

### Unrecognised tax benefits: the stored liability

The **unrecognised tax benefit (UTB)** balance is the accounting reserve for tax positions the company has taken on its returns but is not confident enough to recognise in its accounts. The tabular rollforward is required, and it is a gift to an analyst because it shows the reserve moving.

#### Worked example: what a rising UTB balance is telling you

The same group's UTB rollforward (illustrative):

| | \$m |
| --- | ---: |
| Balance at 1 January | 820 |
| Additions for tax positions taken in the current year | 140 |
| Additions for tax positions of prior years | 60 |
| Reductions for tax positions of prior years | (35) |
| Settlements with taxing authorities | (45) |
| Lapse of the statute of limitations | (30) |
| **Balance at 31 December** | **910** |

Supplementary disclosures: of the \$910m, \$740m would affect the effective tax rate if recognised; accrued interest and penalties are \$150m; and it is reasonably possible that up to \$120m could be resolved within twelve months.

Four readings:

1. **The balance grew 11%** (820 to 910) while pre-tax income grew 4%. The company is taking positions faster than it is resolving them.
2. **\$140m of additions for current-year positions** means this is not a legacy problem being run off. It is happening now.
3. **The \$910m is about 1.3 years of the \$680m annual foreign rate differential.** That is the internal scale: the company itself thinks a bit over a year's worth of its structural benefit might not survive audit.
4. **The \$150m of accrued interest matters.** Tax disputes accrue interest for the whole period they are open, and in the cases at the end of this article the interest was a third of the final bill.

*The intuition: the UTB balance is the company's own estimate of how much of its tax planning it expects to lose. Track its direction, not just its level.*

One related line to check: whether the company asserts that foreign earnings are **indefinitely reinvested** (the APB 23 assertion under US GAAP; a similar exception exists under IAS 12). If so, it records no deferred tax on repatriating them, and it must disclose either the unrecognised deferred tax or that it is not practicable to estimate it. "Not practicable to estimate" on a very large offshore cash pile is a disclosure you should notice.

### Country-by-country reporting: the map arrives

Until 2016, an outsider had essentially no jurisdiction-level data. BEPS **Action 13**, finalised in October 2015, changed that. Groups with consolidated revenue above **€750m** must file, for every jurisdiction in which they operate: revenue (split between related-party and third-party), pre-tax profit, tax paid, tax accrued, stated capital, accumulated earnings, tangible assets, and **number of employees**.

That last field is why the report matters. It puts profit and headcount in the same table.

CbCR was designed as a confidential filing shared between tax authorities, not a public disclosure. But it has been leaking into the public domain: the EU's public CbCR directive (Directive (EU) 2021/2101) requires in-scope groups to publish jurisdiction-level income tax information for financial years beginning on or after **22 June 2024**; Australia and a growing number of voluntary reporters have gone further; and the OECD publishes anonymised, aggregated statistics from the filings.

![Where value is created versus where profit is booked, by jurisdiction](/imgs/blogs/transfer-pricing-and-offshore-profit-shifting-2.webp)

#### Worked example: the profit-per-employee test

An illustrative country-by-country extract for a group with \$31,000m of revenue:

| Jurisdiction | Revenue (\$m) | Pre-tax profit (\$m) | Tax paid (\$m) | Employees | Profit per employee | Cash ETR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| United States | 12,000 | 900 | 180 | 20,000 | \$45,000 | 20.0% |
| Germany | 6,000 | 250 | 75 | 9,000 | \$27,800 | 30.0% |
| Ireland | 9,000 | 600 | 75 | 1,200 | \$500,000 | 12.5% |
| Bermuda | 4,000 | 3,800 | 0 | 6 | \$633,000,000 | 0.0% |
| **Group** | **31,000** | **5,550** | **330** | **30,206** | **\$184,000** | **5.9%** |

Work the ratios:

- Bermuda holds **68.5%** of the group's pre-tax profit (3,800 ÷ 5,550) and **0.02%** of its employees (6 ÷ 30,206).
- Profit per employee in Bermuda is \$633m. In Germany it is \$27,800. That is a ratio of roughly **23,000 to 1**.
- Bermuda books \$4,000m of revenue. Ask the obvious question: who are the customers in Bermuda? On a population of well under 100,000 people, \$4,000m of genuine third-party sales is not credible, which means this revenue is intra-group — and the CbCR template makes you split related-party from third-party revenue, so you can check.
- The group's cash effective rate is **5.9%**, against statutory rates of 21%, 30% and 12.5% in the places where it actually operates.

None of this is proof of anything unlawful. Bermuda might hold genuinely valuable IP acquired in an arm's-length cost-sharing arrangement, and the group might have a defensible transfer-pricing study for every line. But the table gives you the question to ask, and it quantifies the exposure: if Bermuda's profit were taxed at 15% under a global minimum tax, the group's bill would rise by \$570m a year.

*The intuition: profit and people should live in roughly the same places. Where they do not, the CbCR table tells you by how much.*

## Four tests you can run from public filings

Pulling the analytical thread together, here is the practical routine. None of these is dispositive; two or more, persistent across several years, is a pattern worth writing up.

![Four detection tests an outside analyst can run from public filings](/imgs/blogs/transfer-pricing-and-offshore-profit-shifting-9.webp)

**Test 1 — the effective tax rate gap.** Compute the reported ETR for five consecutive years and compare it to a revenue-weighted average of the statutory rates in the countries where the group actually sells. A gap of a few points is ordinary. A gap of ten or more points, stable across five years, is architecture. Then open the reconciliation and check that the gap is explained by named, persistent lines rather than by "other".

**Test 2 — profit per employee by jurisdiction.** Where CbCR data is public, divide pre-tax profit by headcount for each jurisdiction. Within a single group you would expect variation of maybe five or ten times between a research hub and a call centre. Three orders of magnitude is not a business model; it is an allocation.

**Test 3 — revenue booked where there are no customers.** Compare the geographic revenue split in the segment note (which follows customer location) with the jurisdiction split in the CbCR table or the subsidiary list (which follows legal entity). Where a jurisdiction shows large revenue and the segment note shows no corresponding market, the revenue is being booked through, not in, that country.

**Test 4 — related-party pricing without comparables.** In the related-party note, check three things: the *volume* of related-party transactions relative to total sales, whether the note *names the method* used to price them, and whether it identifies any comparables or independent valuation. A note that discloses a large volume, no method and no comparables is the weakest form of this disclosure and the one most worth pressing management on.

A fifth, softer test: read the subsidiary list, which listed companies file (Exhibit 21 in a US 10-K, or the equivalent schedule elsewhere). Count the entities in jurisdictions with no operations. A group with 60 operating countries and 12 Luxembourg entities is telling you something about how it is financed.

## The rules closing in

The environment described above is not static. It has been narrowing for a decade, and the direction of travel matters more for forecasting than any single rule.

![Timeline of the international tax crackdown, 2013 to 2024](/imgs/blogs/transfer-pricing-and-offshore-profit-shifting-8.webp)

**2013–2015: BEPS.** The OECD and G20 launched the Base Erosion and Profit Shifting project in 2013 and published fifteen final action reports in October 2015. The OECD's own estimate at the time put the annual revenue loss from BEPS at **USD 100–240 billion, equivalent to 4–10% of global corporate income tax receipts**. Academic estimates run higher: Wier and Zucman, extending earlier work with Tørsløv, estimate that roughly **USD 969 billion** of profit was shifted in 2019 — around 37% of multinational profits — costing about **USD 247 billion**, or 10% of global corporate income tax revenue.

**2016–2017: implementation.** The EU adopted ATAD, imposing interest limitation, CFC rules, exit taxation, a general anti-abuse rule and (in ATAD 2) anti-hybrid rules across all member states. The US enacted its 2017 reform: a 21% federal rate, a transition tax on accumulated foreign earnings, GILTI as a minimum tax on offshore intangible income, BEAT as an anti-base-erosion minimum tax on deductible payments to foreign affiliates, and FDII as a carrot for holding IP onshore.

**2021–2024: the global minimum tax.** In October 2021 more than 130 jurisdictions in the OECD/G20 Inclusive Framework agreed a two-pillar solution. **Pillar Two** is the operative one: groups with consolidated revenue above **€750m** must pay an effective rate of at least **15%** in every jurisdiction, with a top-up tax collected if they do not. The EU implemented it through Directive (EU) 2022/2523, adopted on 14 December 2022, applying to fiscal years beginning from 31 December 2023 — in practice, 2024.

Pillar Two changes the calculus in a specific way that is easy to state and easy to underrate: **it does not stop you shifting profit to Bermuda. It stops the shifting from being worth anything.** If profit in a jurisdiction is taxed below 15%, someone — the jurisdiction itself through a qualified domestic top-up tax, or the parent's country through the income inclusion rule, or other countries through the undertaxed profits rule — collects the difference. The structure survives; the benefit does not.

**2025–2026: the floor develops a door.** State this one with a date attached, because it is still moving. Alphabet's Form 10-K for fiscal 2025, filed on 5 February 2026, records that "some countries have already implemented the legislation effective January 1, 2024", that this "did not have a material effect on our income tax provision for the 2025 fiscal year", and — the part that matters — that **in January 2026 the OECD introduced guidance including a "Side-by-Side Safe Harbor" which, if elected, exempts US domestic operations from being taxed by global minimum tax rules**, while *not* exempting foreign subsidiaries from local minimum tax requirements where those are implemented.

So as of August 2026 the honest description of Pillar Two is neither "it fixed profit shifting" nor "it collapsed". It is a 15% floor that binds unevenly: in force in a substantial set of jurisdictions since 2024, carrying a significant carve-out for the US domestic operations of US-parented groups, and still being negotiated. If you are modelling a group's tax rate beyond 2026, the scope of that carve-out is the single assumption most likely to be wrong, and the place to check it is the group's own tax footnote rather than any summary of the rules — including this one.

The practical consequences for an analyst: (1) the foreign rate differential line in the reconciliation should compress for in-scope groups, and if a group's guidance still assumes a 12% ETR in 2027 you should ask why; (2) many groups now have a domestic top-up tax charge appearing as a new reconciling line; and (3) jurisdictions that competed on rate are switching to competing on refundable credits and grants, which land in different places in the income statement.

## Common misconceptions

**"A low effective tax rate means the company is cheating."** No. A low ETR can come from operating in genuinely low-tax countries, from R&D credits, from patent-box regimes a parliament deliberately created, from loss carry-forwards, from a share price that rose between grant and vesting of employee stock. The reconciliation tells you which. The test is not the level of the rate; it is whether the explanation is named, persistent and plausible.

**"Transfer pricing is a loophole."** It is a requirement. Every multinational must price its internal transactions, because every entity files a separate return. There is no version of the tax system without transfer pricing. The contested part is not whether prices exist but how much discretion the arm's-length standard leaves.

**"If the tax authority assessed them, they did something wrong."** An assessment is a claim, not a finding. Of the four best-known European state-aid transfer-pricing cases — the four discussed below — the Commission's decisions against Starbucks, Fiat and Amazon were all annulled by the EU courts. Of those four, only Apple survived to final judgment, and even there the Commission lost at first instance before winning on appeal eight years later. (These were not the only EU tax state-aid cases, and the wider record is mixed rather than uniformly bad for the Commission — but every one of them turned on its own national law, which is exactly why you cannot generalise from a headline.) Write "assessed" or "alleged" until a court says otherwise.

**"Moving profit offshore is the same as hiding it."** Under CbCR, master file and local file rules, tax authorities now receive a jurisdiction-by-jurisdiction map of exactly where profit sits. Most large-scale profit shifting is fully disclosed to the authorities that care. It is not concealment; it is a disagreement about pricing, conducted in the open, with lawyers.

**"Tax avoidance and tax evasion are just different words for the same thing."** They are different legal categories with different consequences: one is a dispute about the correct amount of tax, the other is a crime. Conflating them makes your analysis worse, because you will fail to predict the outcome — avoidance cases end in assessments and settlements, evasion cases end in prosecutions, and the base rates are wildly different.

**"Pillar Two ended profit shifting."** It sets a 15% floor for large groups. It does not equalise rates, does not cover groups under the €750m threshold, is implemented unevenly across jurisdictions, and — following the OECD's January 2026 "Side-by-Side" guidance — carves out the US domestic operations of US-parented groups where elected. It compresses the benefit substantially; it does not remove the incentive.

## How it shows up in real markets

Six documented episodes. In each, note carefully what was *alleged*, what was *assessed*, and what a court actually *ruled* — the distinction is the whole point.

### Apple and the European Commission: €13 billion, three benches, eight years

In May 2013, the US Senate Permanent Subcommittee on Investigations held hearings on Apple's offshore structure. The subcommittee's material described Apple Sales International (ASI), an Irish-incorporated company that recorded roughly **\$74 billion of income between 2009 and 2012** and, on the subcommittee's analysis, paid tax at rates measured in hundredths of a percent. The structure relied on the Irish residence rule described earlier: ASI was incorporated in Ireland but claimed to be tax resident nowhere.

On **30 August 2016**, the European Commission concluded that two Irish tax rulings, from 1991 and 2007, had allowed Apple to allocate almost all of ASI's and Apple Operations Europe's profits to "head offices" that existed only on paper, and that this constituted unlawful state aid. The Commission said ASI's effective corporate tax rate fell from about 1% in 2003 to **0.005% in 2014**, and ordered Ireland to recover up to **€13 billion**, plus interest. Both Apple and Ireland — unusually, the beneficiary and the state that granted the aid — appealed.

On **15 July 2020**, the EU General Court annulled the decision (Cases T-778/16 and T-892/16), holding that the Commission had not proved a selective advantage. The Commission appealed on 25 September 2020; a hearing was held on 23 May 2023. On **10 September 2024**, the Court of Justice of the European Union set the General Court's judgment aside and confirmed the Commission's 2016 decision (Case C-465/20 P).

The numbers are worth taking from Apple's own audited filing rather than from the headlines. Apple's 2024 Form 10-K states that the recovery amount was calculated at **€13.1 billion plus €1.2 billion of interest**, subsequently reduced to an adjusted recovery of **€12.7 billion plus €1.2 billion of interest** after credit for taxes paid to other countries; that the escrow balance, including net unrealised investment gains, stood at **€14.2 billion (\$15.8 billion)** at 28 September 2024; and that Apple recorded a **one-time income tax charge of \$10.2 billion, net** in its fiscal fourth quarter of 2024 — \$15.8 billion payable to Ireland out of escrow, offset by a \$4.8 billion US foreign tax credit and an \$823 million decrease in unrecognised tax benefits. The charge lifted Apple's reported effective tax rate for fiscal 2024 to 24.1%, from 14.7% the year before.

That last detail is the whole series in miniature: a tax dispute that ran for eight years arrived in the income statement as a single line, and it moved the effective rate by nearly ten points.

The lesson is not that Apple was found to have done anything criminal; no such allegation was ever made. It is that the same set of facts produced opposite answers at two levels of the same court system, eight years apart, and that a company can carry a thirteen-billion-euro contingency for a decade on a question of law nobody could confidently answer.

### Starbucks, Fiat and Amazon: the Commission's losing streak

The Apple case was one of four. On 21 October 2015, the Commission decided that the Netherlands had granted unlawful aid to **Starbucks** and that Luxembourg had done the same for **Fiat Chrysler Finance Europe**, in each case worth roughly €20–30 million. On 4 October 2017 it decided that Luxembourg had granted **Amazon** approximately €250 million.

All three fell apart:

- **Starbucks**: the General Court annulled the decision on **24 September 2019** (Cases T-760/15 and T-636/16), finding the Commission had not demonstrated an economic advantage. The Commission did not appeal.
- **Fiat**: the General Court upheld the decision in 2019, but the Court of Justice set that aside and annulled the Commission's decision on **8 November 2022** (Cases C-885/19 P and C-898/19 P). The reasoning matters: the arm's-length principle used to test for advantage must be derived from the member state's own national law, not from a free-standing EU version of the OECD standard.
- **Amazon**: the General Court annulled the decision on 12 May 2021, and the Court of Justice dismissed the Commission's appeal on **14 December 2023** (Case C-457/21 P), applying the Fiat reasoning.

For an analyst, this is a caution about probability. The headlines when these decisions land are uniformly "company ordered to pay X". Three of the four Xs never got paid. If you are modelling a tax contingency, model the litigation, not the press release.

### Caterpillar and the Swiss parts structure

In **March 2014** the US Senate Permanent Subcommittee on Investigations published a report on Caterpillar's offshore tax strategy and held a hearing on 1 April 2014. The subcommittee's finding was that Caterpillar had restructured its non-US replacement-parts business so that profits were booked in a Swiss subsidiary, **Caterpillar SARL (CSARL)**, which had negotiated an effective Swiss rate of roughly **4% to 6%**. The subcommittee said the arrangement moved about **\$8 billion** of profit offshore between 1999 and 2012 and deferred or avoided roughly **\$2.4 billion** of US tax, and noted that Caterpillar paid its adviser approximately \$55 million for the work.

Critically, the subcommittee did not allege illegality. The parts themselves largely never touched Switzerland; what moved was title, and the question was whether the Swiss entity's return matched its functions.

The dispute ran for another eight years. Caterpillar's 2022 Form 10-K records that on **8 September 2022** it reached a settlement with the IRS resolving all issues for tax years **2007 through 2016**, without any penalties. The company says it "vigorously contested" the IRS's application of the substance-over-form and assignment-of-income doctrines, and that the settlement "does not include any increases to tax in the United States based on those judicial doctrines and does not include any penalties". The final tax assessed was **\$490 million** for the ten-year period, paid primarily in 2022 along with associated interest of **\$250 million** — about \$740 million in total.

Four things to take from the outcome. The assessed amount was a fraction of the \$2.4 billion the Senate had estimated. The interest was a third of the bill, because the dispute ran a decade. The absence of penalties, and the explicit exclusion of any tax increase resting on the judicial doctrines, is the tell that this resolved as a pricing disagreement rather than as misconduct. And — the point that connects back to the reserve analysis above — Caterpillar disclosed that the settlement "was within the total amount of gross unrecognized tax benefits", so it actually recorded a **\$41 million discrete tax benefit** in 2022 on settling. The company had reserved for more than it ended up paying. That is the unrecognised tax benefit balance doing exactly the job it exists to do, and it is why tracking that balance is worth your time.

### Google's Dutch conduit, in numbers

Because Dutch companies file public accounts, the Double Irish could be observed from outside. Reuters, reporting from the filings of Google Netherlands Holdings BV, reported that the entity routed **€19.9 billion** to Bermuda in 2017 and **€21.8 billion** in 2018. Google confirmed at the end of 2019 that it would wind up the licensing structure, citing the change in US tax law and the Irish rule change.

The value of this case is methodological: it shows that public filings in *any* jurisdiction in the chain can expose the whole chain. If you want to understand a group's structure, the parent's 10-K is often less informative than the local statutory accounts of a subsidiary in a country with public filing requirements.

### Chevron in Australia: pricing the intra-group loan

The clearest judicial statement on intra-group debt came from Australia. Chevron Australia Holdings had borrowed from a US group finance company under a credit facility, and the interest rate reflected the borrower's standalone credit standing without security or covenants.

On **21 April 2017** the Full Federal Court of Australia dismissed Chevron's appeal (*Chevron Australia Holdings Pty Ltd v Commissioner of Taxation* [2017] FCAFC 62). The court's reasoning was commercial: no independent lender would have advanced that much on those terms without security or covenants, and no independent borrower in a group would have ignored the fact that it could obtain a parent guarantee. Pricing the loan as if the borrower were an orphan overstated the arm's-length interest rate. Chevron subsequently discontinued its High Court appeal; the assessed amount was reported at roughly **A\$340 million** including penalties and interest.

This is the case that made intra-group loan pricing a first-order issue globally, and it is why every large group now benchmarks internal interest rates with the same care it applies to goods.

### Glencore in Australia: when the taxpayer wins

The counterweight. The Australian Tax Office challenged the price at which an Australian Glencore subsidiary sold copper concentrate to its Swiss parent, arguing the pricing formula was not arm's length.

The Federal Court found for Glencore in 2019. On **6 November 2020** the Full Federal Court allowed the Commissioner's appeal only in part, and only on a freight issue for one year. On **21 May 2021** the High Court of Australia refused the Commissioner special leave to appeal, with Chief Justice Kiefel stating that no question of principle sufficient to warrant a grant of special leave arose.

The reasoning that carried the day was that the arm's-length test asks what independent parties *could* reasonably have agreed, not what the tax authority thinks would have been optimal — and a pricing formula that a real party might rationally have accepted survives even if a different formula would have produced more Australian tax. That is the strongest statement available of why "the price could have been higher" is not, on its own, an argument.

## What to do with this

If you are reading a set of accounts, four concrete habits:

1. **Always open the tax rate reconciliation.** It takes ninety seconds and it quantifies the offshore structure in dollars per year. Write down the foreign rate differential and treat it as an earnings-at-risk number.
2. **Track the unrecognised tax benefit balance across years**, not just its level. Direction is the signal. An accelerating balance with large current-year additions means the company is still writing positions it does not fully believe in.
3. **Get the jurisdiction map**, from public CbCR where it exists, from the subsidiary exhibit where it does not. Put profit and headcount in the same table and look at the ratio.
4. **Use the right word.** Avoidance, evasion, accounting fraud, and "the tax authority disagrees about a price" are four different claims. Most of what you will find is the fourth.

And if you are trying to forecast rather than diagnose: the structural trade here is that a decade of low-tax earnings is being repriced toward a 15% floor for large groups, unevenly, over several years. The companies most exposed are the ones whose reconciliations show the largest and most persistent foreign rate differentials, and whose CbCR tables show profit in jurisdictions with statutory rates below 15%. That exposure is fully disclosed, sitting in a footnote most people skip.

*This is educational analysis of how financial statements and tax structures work, not investment or tax advice.*

## Sources & further reading

**Primary legal and institutional texts**

- OECD, *Transfer Pricing Guidelines for Multinational Enterprises and Tax Administrations* (2022 edition) — the arm's-length principle and the five methods.
- OECD Model Tax Convention, Article 9 (Associated Enterprises); US Internal Revenue Code section 482 and the regulations thereunder.
- OECD/G20 Base Erosion and Profit Shifting Project, final reports, October 2015 — in particular Actions 4 (interest deductions), 8–10 (aligning transfer pricing with value creation, DEMPE) and 13 (country-by-country reporting, master file, local file; €750m threshold).
- Council Directive (EU) 2016/1164 (ATAD) and Council Directive (EU) 2017/952 (ATAD 2) — interest limitation, CFC rules, GAAR, anti-hybrid rules.
- Council Directive (EU) 2022/2523, adopted 14 December 2022 — the EU implementation of Pillar Two; 15% minimum effective rate, €750m revenue threshold, fiscal years from 31 December 2023.
- Directive (EU) 2021/2101 — EU public country-by-country reporting, for financial years beginning on or after 22 June 2024.
- FASB ASC 740-10 (originally FIN 48) and IFRIC 23 — recognition and measurement of uncertain tax positions; IAS 12 and US GAAP requirements for the tax rate reconciliation.
- HMRC, *Transfer pricing and Diverted Profits Tax statistics 2024 to 2025*, published 11 March 2026 — DPT applied to profits arising from 1 April 2015 at a rate of 31% from 1 April 2023, and is repealed and replaced by the Unassessed Transfer Pricing Profits (UTPP) rules for accounting periods beginning on or after 1 January 2026; transfer-pricing yield £3,387m in 2024–25; DPT £94m in 2024–25 and over £10.5bn secured since introduction; 26 advance pricing agreements agreed at an average of 43.9 months.

**Cases and investigations**

- European Commission decision of 30 August 2016 on state aid granted by Ireland to Apple (SA.38373); General Court judgment of 15 July 2020, Cases T-778/16 and T-892/16; Court of Justice judgment of 10 September 2024, Case C-465/20 P.
- European Commission decisions of 21 October 2015 (Starbucks/Netherlands; Fiat/Luxembourg) and 4 October 2017 (Amazon/Luxembourg); General Court judgment of 24 September 2019 in Starbucks (T-760/15, T-636/16); Court of Justice judgment of 8 November 2022 in Fiat (C-885/19 P, C-898/19 P); Court of Justice judgment of 14 December 2023 in Amazon (C-457/21 P).
- US Senate Permanent Subcommittee on Investigations, *Offshore Profit Shifting and the U.S. Tax Code — Part 2 (Apple Inc.)*, hearing of 21 May 2013.
- US Senate Permanent Subcommittee on Investigations, *Caterpillar's Offshore Tax Strategy*, report and hearing of 1 April 2014.
- *Chevron Australia Holdings Pty Ltd v Commissioner of Taxation* [2017] FCAFC 62 (Full Federal Court of Australia, 21 April 2017).
- *Commissioner of Taxation v Glencore Investment Pty Ltd* — Federal Court (2019); Full Federal Court, 6 November 2020; High Court of Australia special leave refused, 21 May 2021.

**Company filings (the figures quoted for Apple, Caterpillar and Alphabet come from these, not from press reports)**

- Apple Inc., Form 10-K for the fiscal year ended 28 September 2024, Note 7 "Income Taxes" — the European Commission State Aid Decision of 30 August 2016 on tax opinions of 1991 and 2007 covering June 2003 to December 2014; recovery calculated at €13.1bn plus €1.2bn interest, adjusted to €12.7bn plus €1.2bn; escrow balance €14.2bn / \$15.8bn; General Court annulment 15 July 2020; Commission appeal 25 September 2020; hearing 23 May 2023; ECJ setting aside the General Court judgment and confirming the 2016 decision on 10 September 2024; one-time income tax charge of \$10.2bn net; effective tax rate 24.1% in fiscal 2024 against 14.7% in fiscal 2023. The filing also notes that Irish legislative changes effective January 2015 eliminated the application of the tax opinions from that date forward.
- Caterpillar Inc., Form 10-K for the year ended 31 December 2022, Note 6 and the associated critical audit matter — settlement with the IRS on 8 September 2022 resolving all issues for tax years 2007 through 2016 without penalties; final tax assessed \$490m plus associated interest of \$250m; no increases to US tax based on the substance-over-form or assignment-of-income doctrines; settlement within the total gross unrecognised tax benefits, producing a \$41m discrete tax benefit in 2022.
- Alphabet Inc., Form 10-K for the year ended 31 December 2025, filed 5 February 2026 — status of the 15% global minimum tax, including that some countries implemented it effective 1 January 2024 with no material effect on Alphabet's 2025 provision, and that in January 2026 the OECD introduced guidance including a "Side-by-Side Safe Harbor" exempting US domestic operations from global minimum tax rules if elected, while not exempting foreign subsidiaries from local minimum tax requirements.

**Estimates of scale**

- OECD, BEPS project materials: annual revenue loss of USD 100–240 billion, equivalent to 4–10% of global corporate income tax revenues.
- L. Wier and G. Zucman, and earlier work with T. Tørsløv, on the missing profits of nations: approximately USD 969 billion of profit shifted in 2019 (about 37% of multinational profits), implying a revenue loss near USD 247 billion, or 10% of global corporate income tax revenue.

**Related posts in this series**

- [Related-party transactions and self-dealing](/blog/trading/forensic-accounting/related-party-transactions-and-self-dealing)
- [Off-balance-sheet financing and special purpose entities](/blog/trading/forensic-accounting/off-balance-sheet-financing-and-special-purpose-entities)
- [Shell companies, reverse mergers and how fraud gets listed](/blog/trading/forensic-accounting/shell-companies-reverse-mergers-and-how-fraud-gets-listed)
- [The footnotes and MD&A: where the bodies are buried](/blog/trading/forensic-accounting/the-footnotes-and-mda-where-the-bodies-are-buried)
- [Cookie-jar reserves and big-bath accounting](/blog/trading/forensic-accounting/cookie-jar-reserves-and-big-bath-accounting)
- [Round-tripping and fabricated revenue](/blog/trading/forensic-accounting/round-tripping-and-fabricated-revenue)
- [Reading the cash flow statement: why cash beats net income](/blog/trading/forensic-accounting/reading-the-cash-flow-statement-why-cash-beats-net-income)

*Illustrative arithmetic throughout — the widget chain, the royalty strip, the thin-capitalisation comparison, the Double Irish walkthrough, the tax rate reconciliation, the unrecognised tax benefit rollforward and the country-by-country extract — uses invented round numbers to show mechanics. Every figure attributed to a company, court, regulator or research paper is sourced above and dated.*

*One closing point of framing, because this article sits inside a series about fraud. The structures described here — migrating intellectual property, intra-group debt, service fees, principal structures, and the Double Irish itself — were lawful tax planning, disclosed to the authorities and litigated in open court. No company named in this article has been found to have committed accounting fraud in connection with its transfer pricing, and none has been alleged to. Transfer pricing belongs in a forensic accounting series for one reason only: it is where a company's tax reserves, its disclosures and its earnings can be misstated. The planning is not the offence. The misreporting of it would be.*
