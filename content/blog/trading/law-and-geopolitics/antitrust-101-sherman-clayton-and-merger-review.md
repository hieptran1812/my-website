---
title: "Antitrust 101: Sherman, Clayton, and how merger review gates M&A"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "How antitrust law decides which mergers close, which monopolies stand, and how much pricing power a firm may keep — and how to read the HHI, the HSR process, and the review odds to handicap a deal."
tags: ["antitrust", "regulation", "mergers", "hhi", "merger-review", "hsr", "monopoly", "competition-law", "deal-risk", "event-driven", "geopolitics"]
category: "trading"
subcategory: "Law & Geopolitics"
author: "Hiep Tran"
featured: true
readTime: 44
---

> [!important]
> **TL;DR** — Antitrust law sets the rules for how much market power a company is allowed to build and keep, so it directly decides which mergers close, which monopolies survive, and how durable a dominant franchise's pricing power really is.
>
> - The system rests on three statutes: the **Sherman Act** (bans monopolizing conduct and conspiracies), the **Clayton Act** (blocks mergers that "substantially lessen competition"), and the **FTC Act** (a catch-all for unfair methods). Two agencies — the **DOJ Antitrust Division** and the **FTC** — enforce them.
> - Big merger? You must file a **Hart-Scott-Rodino (HSR)** notice and wait. Most deals clear the **30-day** waiting period. The danger is the **second request** — a document demand that adds **6 to 18 months** and is where deal value gets destroyed.
> - The case turns on **market definition** and the **HHI** (Herfindahl-Hirschman Index), a concentration score that *squares* each firm's share. A merger of a 30%-share and 20%-share firm in the same market raises the HHI by **2 × 30 × 20 = 1,200 points** — deep into "presumed harmful" territory.
> - The one number to remember: **monopoly itself is not illegal — only the conduct or the merger that creates it.** And the agencies *clear the large majority of deals;* fewer than ~3% draw a serious challenge. Read the HHI and the second-request signal to handicap which deal is the exception.

On August 5, 2024, a federal judge ruled that Google had illegally monopolized the market for general internet search. The case — *United States v. Google* — had been filed almost four years earlier under a statute written in **1890**, the Sherman Antitrust Act. The ruling did not break up Google, and it did not impose a fine. It declared a legal fact: that Google's conduct, including paying billions of dollars a year to be the default search engine on phones and browsers, was an illegal *maintenance* of a monopoly. Alphabet's stock barely moved that day — the market had long since priced a high probability of the ruling — but the *remedies* phase that followed would decide whether Google had to share its data, divest its Chrome browser, or merely change its default contracts. Tens of billions of dollars of franchise value hung on the answer.

That is antitrust law doing what it always does: setting the rules for how much market power a company is allowed to build, and how much it is allowed to keep. Those rules are not a footnote for lawyers. They decide whether two companies can combine, whether a dominant firm can charge what it likes, and therefore what a deal is worth and how durable a franchise is. When a merger is announced, the single biggest question for a trader is not "is this a good business combination" — it is "will the antitrust agencies let it happen, and on what terms." When you own a dominant company, the biggest tail risk to its multiple is often not a competitor — it is a regulator deciding the moat is illegal.

This post builds the antitrust machine from zero: the three statutes, the two agencies, the review process, and the one number — the HHI — that does most of the work. Then it turns to the payoff: how to read the concentration math and the procedural signals to handicap whether a deal clears, gets a remedy, or dies. We trace [how law moves markets through the full transmission chain](/blog/trading/law-and-geopolitics/how-law-moves-markets-the-transmission-chain) elsewhere; here we zoom in on the link from competition law to deal value and franchise durability.

![Three antitrust statutes feeding into what each polices then into agency enforcement and the market value at stake](/imgs/blogs/antitrust-101-sherman-clayton-and-merger-review-1.png)

## Foundations: the three statutes, the two agencies, and the words that decide a case

Antitrust law sounds technical, but the core idea is simple and old. A market works — sets fair prices, rewards better products, pushes firms to innovate — only when firms *compete*. When competition disappears, the surviving firm can raise prices, cut quality, and stop trying, and customers have nowhere else to go. Antitrust law is the body of rules designed to keep markets competitive: it bans the behaviors and the deals that destroy competition. Everything else is detail. Let us build that detail from zero, because the detail is where the money is.

### The everyday version: the only grocery store in town

Take a town with four grocery stores. They compete on price and quality; if one gouges you, you shop at another. Now the two biggest stores merge, then the merged store buys a third. One store is left standing against it, and then that one closes. The survivor now has no reason to keep prices low, stock fresh produce, or stay open late. It is a **monopoly** — a single seller with no effective competition — and the town pays for it in higher prices and worse service.

Antitrust law is the town's defense against that outcome. It does three things, which map onto the three statutes. It punishes the stores if they secretly agree to fix prices instead of competing (the Sherman Act bans that *conduct*). It can block the mergers *before* they happen if the combination would lessen competition (the Clayton Act gates the *structure*). And it has a flexible backstop to catch new tricks the older laws did not foresee (the FTC Act). Hold this grocery-store picture in mind; every legal term below is just a precise version of it.

### The Sherman Act (1890): the ban on monopolizing conduct

The **Sherman Antitrust Act**, passed in 1890 during the era of the great railroad and oil "trusts," is the foundation. It has two operative sections.

**Section 1** bans "every contract, combination… or conspiracy, in restraint of trade." In plain terms: competitors cannot agree with each other to kill competition. The clearest example is **price-fixing** — rivals secretly agreeing to charge the same high price instead of undercutting each other. Section 1 also reaches bid-rigging, market-allocation deals ("you take the East Coast, I'll take the West"), and group boycotts.

**Section 2** bans **monopolization** — but here is the single most misunderstood point in all of antitrust: *being a monopoly is not illegal.* Section 2 bans the *act* of acquiring or maintaining a monopoly through improper conduct, not the mere fact of having a large market share. A company that wins 90% of a market by building a better product, charging less, and serving customers well has done nothing wrong. A company that wins or keeps that share by sabotaging rivals, signing exclusionary contracts, or buying up every nascent competitor has violated Section 2. The Google search ruling was a Section 2 case: the court found not that Google was big, but that its *default-payment conduct* illegally maintained the monopoly. The distinction — big versus bad conduct — is the hinge of the whole field.

### The Clayton Act (1914): the merger gate

The Sherman Act punishes bad conduct *after* it happens. Congress soon realized that was too late: by the time a monopoly forms and abuses its power, the damage is done and unwinding it is nearly impossible. So in 1914 it passed the **Clayton Antitrust Act** to stop anticompetitive market structures *before* they form.

The crucial provision is **Section 7**, which prohibits any merger or acquisition whose effect "may be **substantially to lessen competition** or to tend to create a monopoly." Read that standard carefully, because it does the heavy lifting in every merger fight:

- It is **forward-looking and probabilistic** — "*may be* substantially to lessen competition." The agencies do not have to prove harm has occurred; they have to show a *reasonable probability* that it will. This is why merger review is a prediction exercise, and why it is so contestable.
- "**Substantially**" is a threshold, not a switch. A deal that trims competition a little is fine; a deal that meaningfully reduces it is not. The whole question is *how much* — which is exactly what the HHI tries to measure.

The Clayton Act also bans certain kinds of **price discrimination** (the Robinson-Patman Act amended this in 1936), **exclusive-dealing** and **tying** arrangements where they lessen competition, and interlocking directorates. But for an investor, Section 7 — the merger gate — is the one that moves deal prices.

### The FTC Act (1914): the flexible backstop

The same year, Congress created the **Federal Trade Commission** and gave it the **FTC Act**, whose Section 5 bans "**unfair methods of competition**" and "unfair or deceptive acts or practices." This is deliberately broad. It lets the FTC reach conduct that the older, more specific statutes might not clearly cover — novel exclusionary tactics, deceptive practices, and emerging harms. Think of it as the catch-all that keeps antitrust from ossifying around 1890s and 1910s fact patterns.

### The two cops: DOJ Antitrust Division vs. the FTC

The United States is unusual in having *two* federal antitrust enforcers with overlapping authority:

- The **Antitrust Division of the Department of Justice (DOJ)** — part of the executive branch, can bring *criminal* charges (price-fixing executives go to prison), and litigates in federal court.
- The **Federal Trade Commission (FTC)** — an independent agency with civil authority, can litigate in federal court *and* in its own in-house administrative process.

When a deal is announced, the two agencies "clear" it between themselves — one takes the lead based on industry expertise (the DOJ historically handles telecom, airlines, banking; the FTC handles healthcare, retail, tech-adjacent consumer markets). For a trader, *which* agency reviews a deal can matter: the agencies differ in posture, speed, and litigation record, and a deal that one would wave through, the other might fight.

### Per-se illegal vs. rule-of-reason: the two ways courts judge conduct

Not all anticompetitive conduct is judged the same way. Courts use two very different standards, and knowing which applies tells you how a case will likely go.

- **Per-se illegal** conduct is so obviously harmful that no defense is allowed. The classic per-se categories are *horizontal price-fixing*, *bid-rigging*, and *market allocation* among competitors. If the prosecution proves the agreement happened, the defendant loses — there is no "but it was actually good for consumers" argument. These are the cases that send executives to prison.
- **Rule-of-reason** conduct is judged on its actual competitive effect. The court weighs the procompetitive benefits against the anticompetitive harms in the specific market. Most conduct — vertical restraints, most exclusive dealing, and crucially almost all *merger* analysis — falls under the rule of reason. This is fact-intensive, expensive, and uncertain, which is exactly why a contested merger review is a multi-year, multi-million-dollar fight.

There is one more wrinkle that trips up newcomers: the **"quick look"** middle ground. Some conduct is not quite obviously illegal enough for the per-se label, yet so likely to be harmful that a court will not require the full, years-long rule-of-reason inquiry. In a quick-look case the burden effectively shifts to the defendant to justify the restraint. For an investor, the practical signal is simple: the *more* a court treats conduct as per-se or quick-look, the *higher* the odds the defendant loses, and the faster the case resolves; the *more* it is full rule-of-reason, the more the outcome depends on a sprawling, uncertain economic fight that can swing either way.

### What "monopoly power" actually means in a Section 2 case

It is worth being precise about the Section 2 monopolization standard, because dominant-franchise risk is one of the biggest applications of antitrust for equity investors. A Section 2 violation has two elements: (1) the firm possesses **monopoly power** in a properly defined market, and (2) it has engaged in **exclusionary or anticompetitive conduct** to acquire or maintain that power, as distinct from growth on the merits.

Monopoly power is usually inferred from a very high market share — courts have historically treated shares above roughly **70%** as strong evidence of monopoly power, shares in the **50-70%** range as ambiguous, and shares below **50%** as rarely sufficient on their own. But share is only the screen; the case turns on the *conduct*. Refusing to deal with rivals, exclusive contracts that lock up distribution, predatory pricing (selling below cost to drive out a competitor, then raising prices once it is gone), tying a monopoly product to a second product, and serial "killer acquisitions" of nascent threats are the classic exclusionary theories. The defense is always the same: "we won by being better, and our conduct is normal competition." That is why the Section 2 case against a dominant platform is so consequential and so contested — the entire fight is over whether the moat was *built* or *abused*, and the answer re-rates the franchise. A monopolization finding does not automatically break a company up, but it opens the door to structural remedies, and the *possibility* of a forced divestiture is what compresses a dominant firm's multiple long before any court orders one.

### The consumer-welfare standard — and the challenge to it

For roughly four decades, US antitrust has been anchored to the **consumer-welfare standard**: the idea, popularized by Robert Bork and the "Chicago School" in the late 1970s, that the goal of antitrust is to protect *consumers* — chiefly through low prices and high output — rather than to protect *competitors* or to limit bigness for its own sake. Under this standard, a merger is fine if it does not harm consumers, even if it makes a company huge.

This standard is now genuinely contested. A movement sometimes called the **New Brandeisians** (after Justice Louis Brandeis, who distrusted concentrated economic power) argues that the consumer-welfare standard is too narrow — that it ignores harms to workers, suppliers, innovation, and the political economy, and that "low prices today" can mask a monopoly being built (think of a platform that subsidizes prices to gain dominance, then extracts later). The 2021-2024 leadership at the FTC and DOJ pushed this more aggressive, structure-focused view. For an investor, the *standard itself drifting* is a regulatory risk: a deal that would have cleared under a pure consumer-welfare reading may draw a challenge under a broader one. We unpack this shift in [big tech antitrust and the New Brandeisians](/blog/trading/law-and-geopolitics/big-tech-antitrust-the-new-brandeisians).

### Horizontal vs. vertical mergers

Two firms can combine in two fundamentally different ways, and the law treats them differently.

- A **horizontal merger** combines *direct competitors* in the same market — two grocery chains, two airlines on the same route, two cloud providers. This *directly* removes a competitor and raises concentration. Horizontal deals get the most scrutiny because the harm is the most obvious.
- A **vertical merger** combines firms at *different stages of the same supply chain* — a manufacturer buying its parts supplier, a studio buying a distributor. No direct competitor is removed, but the merged firm might **foreclose** rivals: a chipmaker that buys a key materials supplier could starve competing chipmakers of that input, or raise its price. Vertical theory is subtler and historically cleared more easily, but the 2020s brought far more aggressive vertical enforcement.

We will return to these with a worked HHI computation and a side-by-side comparison below.

With the vocabulary built, we can follow a real deal through the machine.

## The deal-review pipeline: from signing to clearance, remedy, or block

A merger above a size threshold does not just "happen." It runs a legal gauntlet, and every stage is a place where deal value can be created or destroyed. This is the single most important process for anyone trading merger risk to understand.

![HSR merger review pipeline from signing the deal through filing waiting period second request and the final outcome](/imgs/blogs/antitrust-101-sherman-clayton-and-merger-review-2.png)

### Step 1: the Hart-Scott-Rodino (HSR) filing

The **Hart-Scott-Rodino Antitrust Improvements Act of 1976** created the modern pre-merger notification system. Its logic is preventive: rather than let a merger close and then try to unscramble the egg, HSR forces large deals to *notify the agencies in advance and wait* before closing, giving the government a chance to review and, if needed, sue to stop it.

If a transaction exceeds a **size-of-transaction threshold** — adjusted annually for inflation, around **\$120 million** in the mid-2020s — both the buyer and the seller must file an HSR notification with the FTC and DOJ, pay a filing fee (which scales with deal size, from tens of thousands to over a million dollars for the largest deals), and **wait**. They cannot legally close until the waiting period expires or is terminated. The point is the *wait*, not the paperwork: HSR buys the government time to look before the deal is consummated.

### Step 2: the 30-day waiting period

After filing, a standard **30-day waiting period** begins (15 days for an all-cash tender offer or a bankruptcy sale). During this window the agencies do their triage: they assess the parties' market overlap, run the HHI math, and decide whether the deal raises any competitive flag. For the overwhelming majority of deals — combinations in unconcentrated markets, deals between firms that do not compete — the answer is "no concern," and the waiting period simply expires (or the agency grants **early termination**, letting the parties close ahead of schedule). The deal clears. This is the normal path, and it is worth repeating: **most deals clear the initial wait.** The drama is the exception.

### Step 3: the second request — where deals go to die

If a deal *does* raise a flag, the reviewing agency issues a **second request** (formally, a "Request for Additional Information and Documentary Material") before the 30 days expire. This is the moment everything changes. A second request is a sweeping, burdensome demand for documents, data, emails, and custodian files going back years. Complying can cost the parties tens of millions of dollars and take many months.

Critically, the second request **restarts the clock**: the parties cannot close until they have "substantially complied" and a *new* waiting period (typically 30 more days) runs after that. In practice, a second request adds **6 to 18 months** to a deal's timeline and signals that the agency is seriously investigating. The mere issuance of a second request is the loudest procedural signal in merger arbitrage. Only a small fraction of HSR filings draw a second request — historically on the order of **2-4%** — but those are the deals that get challenged, remedied, or abandoned.

### Step 4: clearance, consent decree, or challenge

After the investigation, the agency reaches one of three outcomes:

1. **Clearance.** The agency closes the investigation and the deal proceeds. The overhang lifts and, in merger arb, the spread collapses to zero.
2. **A consent decree (a remedy).** The agency will allow the deal *if* the parties fix the competitive problem. The standard fix is a **divestiture** — the merged firm must sell off the overlapping business (a brand, a set of stores, a product line) to a credible buyer who will keep competing. A weaker fix is a **behavioral remedy** — a binding promise not to misuse the new market power (e.g., not to discriminate against rivals, to keep an input available on fair terms). Behavioral remedies are harder to police and have fallen out of favor with structure-focused enforcers, who prefer a clean divestiture.
3. **A challenge (sue to block).** If the agency concludes no remedy can fix the harm, it sues in federal court to block the deal. Now a judge decides. Litigation takes a year or more, and many parties **abandon** the deal rather than fight — the uncertainty, the cost, and the break-up clock kill the strategic rationale even before a verdict.

There is a contractual layer riding on top of all of this that an investor must read. Merger agreements specify *who bears the antitrust risk* through several mechanisms. The **outside date** (or "drop-dead date") is the deadline by which the deal must close or either party can walk — a long review that pushes past the outside date can kill a deal even if it would eventually have cleared. **Break fees** (also called termination fees) are cash penalties: a *reverse* break fee is what the acquirer pays the target if regulators block the deal, and a large reverse break fee signals the acquirer is confident enough to put real money behind the antitrust risk. A **"hell-or-high-water" clause** obligates the acquirer to do *whatever* the regulators demand — divest anything, agree to any condition — to get the deal done; its presence is a strong signal that the parties expect a fight and are committed to winning it, while its *absence* means the acquirer reserves the right to walk if the required remedy is too painful. When you read a merger agreement, the size of the reverse break fee and the strength of the regulatory-efforts clause tell you how the *parties themselves* are handicapping the antitrust risk — often the single best outside estimate available.

The whole process is the reason a merger announcement is not a done deal. The price the acquirer is paying is a *promise* contingent on clearing this gauntlet. The job of the merger-risk trader is to handicap which exit the deal takes — and the math that drives it is the HHI.

## Market definition and the HHI: the number that decides the case

Everything in merger review turns on two linked questions: *what is the market*, and *how concentrated does the merger make it*. Get the market wrong and the whole analysis collapses; get the concentration math right and you can predict the agency's posture.

### Market definition: the most consequential argument in the case

You cannot measure a firm's market share until you define the *market*. And market definition is where most merger fights are actually won or lost, because the parties and the agency fight bitterly over how *narrowly* or *broadly* to draw the boundary.

The standard tool is the **SSNIP test** (a "Small but Significant Non-transitory Increase in Price," usually 5%). The question: if a hypothetical monopolist of the proposed market raised prices 5%, would enough customers switch to other products that the price hike would be unprofitable? If yes, the market is drawn too narrowly — those substitute products belong inside it. If no, the boundary holds.

This sounds academic, but it is worth billions. If a merger of two craft-beer makers is analyzed in a "craft beer" market, their combined share is huge and the deal looks anticompetitive. If it is analyzed in an "all beer" or "all alcoholic beverages" market, their share is tiny and the deal looks harmless. The parties will argue for the broadest possible market; the agency, for the narrowest plausible one. *Whoever wins the market definition usually wins the case.* As an investor handicapping a deal, your first question should always be: how narrowly can the agency credibly define this market, and what is the combined share there?

Market definition has a *geographic* dimension too, not just a product one. Two hospital systems that do not overlap anywhere — one serves a city in the east, the other a city in the west — can merge freely, because the relevant geographic market for hospital care is *local*: a patient in one city will not drive across the country for routine surgery. But two hospitals in the *same* metropolitan area, even if they are nominally "small," can have an enormous combined *local* share and face a block. This is why supermarket, hospital, and gas-station mergers are analyzed store-by-store and metro-by-metro: the national share can look modest while a dozen specific local markets each cross the structural presumption. The divestiture remedies in these deals are correspondingly local — sell the stores in *these* twelve overlapping towns — which is exactly why reading the *local* overlap map, not the national share, is what tells you the likely size of the carve-out.

#### Worked example:

A regulator weighs two market definitions for a proposed deal. Under the broad definition, the merging firms have **8%** and **6%** shares — a combined **14%**, with a ΔHHI of `2 × 8 × 6 = 96`, *below* the 100-point trigger; the deal looks fine. Under the narrow definition — stripping out distant substitutes that the SSNIP test says customers would not actually switch to — the same two firms have **35%** and **25%** shares, a combined **60%**, with a ΔHHI of `2 × 35 × 25 = 1,750`. **That is the entire ballgame:** the same deal is presumptively legal under one market definition and flagrantly illegal under the other. **The intuition: market definition is not a preliminary technicality — it is the lever that moves the combined share and the ΔHHI by an order of magnitude, so the first thing to estimate when handicapping a deal is which definition the agency can credibly defend in court.**

### The HHI: squaring shares to measure concentration

Once the market is defined and shares are assigned, the agencies measure concentration with the **Herfindahl-Hirschman Index (HHI)**. The formula is simple and the *squaring* is the whole point:

```
HHI = (share_1)^2 + (share_2)^2 + ... + (share_n)^2
```

You take each firm's market share as a percentage, square it, and add them up. The index runs from near 0 (perfect competition, thousands of tiny firms) to 10,000 (a pure monopoly, one firm with 100%: 100² = 10,000).

Why square? Because squaring gives *disproportionate weight to large firms*, which is exactly the property you want. A market with one 50%-share firm and fifty 1%-share firms is far more concentrated, in any meaningful sense, than a market with a hundred 1%-share firms — even though the shares "add up" the same way. Squaring captures that: the 50% firm contributes 2,500 to the HHI all by itself, while a 1% firm contributes just 1.

![Matrix showing each firm market share squared and summed into an HHI before and after a merger with the delta computed](/imgs/blogs/antitrust-101-sherman-clayton-and-merger-review-3.png)

The Merger Guidelines define rough concentration zones (the exact thresholds have shifted over time, and the 2023 update tightened them):

- **Unconcentrated:** HHI below ~1,000 — mergers here rarely raise concern.
- **Moderately concentrated:** HHI ~1,000 to ~1,800 — closer scrutiny.
- **Highly concentrated:** HHI above ~1,800 (the 2023 Guidelines set a structural presumption at **1,800**, down from the older 2,500) — mergers that push into or within this zone face a presumption of harm.

#### Worked example:

Take the four-firm market from the figure: Firm A has **30%** of the market, Firm B **20%**, and two rivals C and D have **25%** each. Compute the HHI before any merger by squaring and summing:

```
HHI_before = 30^2 + 20^2 + 25^2 + 25^2
           = 900 + 400 + 625 + 625
           = 2,550
```

This market is already "highly concentrated" (above 1,800). Now suppose A and B — the 30% and 20% firms — propose to merge. The merged firm has **50%** of the market. Recompute:

```
HHI_after  = 50^2 + 25^2 + 25^2
           = 2,500 + 625 + 625
           = 3,750
```

The change — the **ΔHHI** — is `3,750 − 2,550 = 1,200`. There is a beautiful shortcut: the increase from a horizontal merger is always **twice the product of the two merging firms' shares**, `ΔHHI = 2 × share_A × share_B = 2 × 30 × 20 = 1,200`. Under the 2023 Guidelines, a deal that produces a post-merger HHI above 1,800 *and* a ΔHHI above 100 is presumptively illegal. This deal blows through both: a final HHI of 3,750 and a ΔHHI of 1,200. **The intuition: because the HHI squares shares, combining two already-large firms produces an enormous jump — this deal would almost certainly draw a second request and likely a challenge or a forced divestiture.**

The squaring is why two big firms merging is so much more alarming than the same total share spread across small players: the ΔHHI of `2 × 30 × 20 = 1,200` would be only `2 × 5 × 5 = 50` if the merging firms each had 5% — below the 100-point trigger entirely. That is the single most important piece of arithmetic in handicapping a horizontal deal.

### A real, highly concentrated market

To see the HHI on real numbers, look at the global chip-foundry market — the firms that physically manufacture semiconductors. It is one of the most concentrated markets on earth.

![Global chip foundry market share bar chart with TSMC at 64 percent and the computed HHI of 4384](/imgs/blogs/antitrust-101-sherman-clayton-and-merger-review-6.png)

#### Worked example:

With TSMC at roughly **64%**, Samsung **11%**, SMIC **6%**, UMC **5%**, GlobalFoundries **5%**, and others **9%**, the HHI is:

```
HHI = 64^2 + 11^2 + 6^2 + 5^2 + 5^2 + 9^2
    = 4,096 + 121 + 36 + 25 + 25 + 81
    = 4,384
```

An HHI of **4,384** is staggeringly high — well into the "highly concentrated" zone, and most of it (4,096 of 4,384) comes from TSMC alone, because its 64% share gets squared. Now imagine a hypothetical merger of the #2 and #3 players, Samsung and SMIC. The ΔHHI would be `2 × 11 × 6 = 132` — above the 100-point trigger, in an already-hyper-concentrated market. Any such combination would face intense scrutiny (and, given that these are firms in the US, Korea, and China, it would also collide with national-security review). **The lesson: in a market this concentrated, the squaring means even a modest combination of mid-sized players trips the structural presumption — there is simply no room left to combine.** The foundry concentration is itself a geopolitical fault line, which we explore in the chip-war and supply-chain posts.

## Vertical mergers and the foreclosure theory

Horizontal mergers are the easy case: remove a competitor, raise the HHI, presume harm. Vertical mergers are subtler, and the law's treatment of them has swung sharply.

![Matrix comparing horizontal and vertical mergers by what combines how each harms competition and the antitrust theory](/imgs/blogs/antitrust-101-sherman-clayton-and-merger-review-4.png)

A **vertical merger** joins a firm with its supplier or its customer — different links in the same chain, not direct rivals. Because no competitor is removed, the HHI in any single market may not change at all. So where is the harm? The leading theory is **foreclosure**: the merged firm, now owning a critical input *and* competing downstream, can **raise rivals' costs** or cut them off entirely. A chipmaker that buys the only supplier of a key material can starve competing chipmakers of it, or sell it to them at an inflated price, tilting the downstream market in its own favor.

For decades, courts were skeptical of vertical-harm theories — the Chicago School argued that a vertical merger usually just improves efficiency (the "double marginalization" problem: a vertically integrated firm has *less* incentive to gouge, because it eats its own margin). That skepticism produced very few vertical challenges. But the 2020s brought a far more aggressive posture: the agencies challenged vertical deals on foreclosure grounds (the AT&T–Time Warner case, which the government lost; later deals where the threat alone reshaped terms), and the 2023 Merger Guidelines elevated vertical theories. For a trader, the takeaway is that **vertical does not mean safe anymore** — a deal you might once have assumed would sail through because "they don't compete" can now draw a foreclosure challenge.

#### Worked example:

Suppose a downstream device maker with **40%** of its market acquires the supplier of a component used by *all* device makers. The HHI in the device market is unchanged by the deal — no rival device maker disappears. But model the foreclosure: if the merged firm raises the component's price to rivals by enough to add **\$8** of cost to each rival device that sells for **\$200**, that is a **4%** cost increase imposed on 60% of the market. Even with no change in market shares on day one, the agency's theory is that this cost handicap will, over time, let the merged firm win share and raise prices — "substantially lessening competition" in the Clayton Act's forward-looking sense. **The intuition: vertical harm shows up not as a one-day HHI jump but as a slow tilt of the playing field, which is exactly why it is harder to prove and why these cases are so contested.**

## The 2023 Merger Guidelines and the more aggressive posture

The **Merger Guidelines** are not law — they are the agencies' published statement of *how they analyze* mergers, a window into their thinking that courts often find persuasive. They have been revised repeatedly (1968, 1982, 1992, 2010, and 2023), and each revision is a signal of regulatory regime change that investors should read like a policy statement from a central bank.

The **2023 Guidelines** (issued jointly by the DOJ and FTC) marked the most aggressive shift in a generation. The headline changes:

- **Lower structural thresholds.** The presumption of harm now kicks in at a post-merger HHI of **1,800** (down from 2,500) with a ΔHHI above 100, *or* at a combined market share above 30% with a ΔHHI above 100. More deals fall into the "presumed illegal" bucket.
- **Explicit attention to non-price harms** — effects on workers (labor-market concentration / monopsony), innovation, and potential competition — reflecting the New Brandeisian critique of the pure consumer-welfare standard.
- **Scrutiny of "serial acquisitions" and roll-ups** — a pattern of many small acquisitions that individually clear but cumulatively build dominance (a recurring concern with private-equity roll-ups and big-tech "acqui-hire" sprees).
- **Tougher vertical and platform theories**, as discussed above.

For an investor, a Guidelines revision is a *re-rating event* for deal risk across whole sectors. After 2023, the market's implied probability of approval for borderline horizontal and platform deals fell, merger-arb spreads on contested deals widened, and some strategic acquirers simply stopped pursuing deals they judged unwinnable. Reading the regulatory posture — not just the static law — is essential, which is why we treat [regulatory risk as an asset-pricing factor](/blog/trading/law-and-geopolitics/regulatory-risk-as-an-asset-pricing-factor) in its own post.

## How a challenge or a long review destroys deal value and timelines

Here is the link from law to dollars, and it is the heart of why this matters for trading. When a merger is announced, the target's stock typically jumps *toward* — but not all the way to — the offer price. The remaining gap, the **merger-arb spread**, is the market's compensation for the risk that the deal does not close. Antitrust is one of the biggest drivers of that risk.

A deal can be destroyed by antitrust in two ways, and both cost real money:

1. **The block.** If the agency sues and wins (or the parties abandon), the deal dies. The target, which had risen toward the offer price on the announcement, *falls back toward its standalone value* — the price it would trade at with no deal. The gap between the deal price and the standalone price is the loss for anyone who bought the target betting on a close. This is the **downside** in merger arbitrage, and it can be brutal: a target that rose from \$80 to \$98 on a \$100 announced deal falls all the way back to \$80 (or lower, if the failed process revealed problems) when the deal breaks.

2. **The long review.** Even a deal that *eventually* clears can destroy value through *time*. Merger arb is a money-over-time business: the spread is an annualized return. A deal expected to close in 4 months that instead drags through an 18-month second request earns a fraction of the expected annualized return, ties up capital, and exposes the position to the risk that something else goes wrong in the interim. A second request alone — even before any block — re-rates the deal's expected timeline and crushes the annualized spread.

#### Worked example:

A target trades at a standalone price of **\$80**. An acquirer offers **\$100** in cash. On announcement the target jumps to **\$96** — a **\$4** spread to the \$100 deal price. That \$4 is the market's price for the deal risk. Now decompose it with a probability model. Suppose the market thinks the deal has a **90%** chance of closing at \$100 and a **10%** chance of breaking, in which case the target falls back to \$80. The probability-weighted value is:

```
expected value = 0.90 x $100 + 0.10 x $80
               = $90 + $8
               = $98
```

But the stock trades at \$96, not \$98 — because the \$98 is a *future* value and the arb must be discounted for the time and capital tied up. The \$2 gap between the \$98 expected value and the \$96 price is the arb's reward for bearing the risk over the expected holding period. **The intuition: the merger-arb spread is a bet on the antitrust outcome, and you can back out the market's implied probability of approval directly from where the target trades relative to the deal price and the standalone price.** We work the full position-sizing version in [merger arbitrage and trading regulatory deal risk](/blog/trading/law-and-geopolitics/merger-arbitrage-trading-regulatory-deal-risk).

## International review: a deal needs more than one government to say yes

So far we have treated merger review as a US affair. For any large multinational deal, that is a dangerous simplification. A global merger does not need *one* clearance — it needs clearance from *every* jurisdiction where the parties do meaningful business, and a single hostile regulator anywhere can sink the whole deal.

The major review regimes that matter for a large cross-border deal are the United States (DOJ/FTC), the **European Union** (the European Commission's Directorate-General for Competition, "DG COMP"), the **United Kingdom** (the Competition and Markets Authority, the CMA), and **China** (the State Administration for Market Regulation, SAMR). Beyond those, dozens of smaller national regulators may each have a notification requirement. The thresholds, the timelines, and the *standards* differ — the EU has historically been more willing to demand structural remedies, the UK's CMA became notably more assertive post-Brexit (it can review global deals with even a modest UK nexus), and China's SAMR review has at times been used as a geopolitical lever, slow-walking deals involving US firms during periods of trade tension.

The practical consequence for an investor is **the weakest-link problem**: the probability that a global deal closes is the probability that it clears the *toughest* jurisdiction, not the average one. A deal can sail through the US and the EU and still die in the UK, or stall indefinitely in China. When you handicap a multinational merger, you must map *all* the required clearances and identify which regulator is the binding constraint. Several high-profile technology and semiconductor deals in the 2020s were abandoned not because the US blocked them but because one foreign regulator's review dragged past the deal's outside date or signaled a block. The same logic links to national-security screening (CFIUS in the US, and its analogs abroad), where the review is about who *owns* the asset rather than competition — a separate gauntlet that a cross-border deal must also clear.

#### Worked example:

A global deal must clear four jurisdictions. From the concentration math and each regulator's posture, you assign independent approval probabilities: US **90%**, EU **85%**, UK **80%**, China **75%**. If the reviews were independent, the probability the deal clears *all four* is the product:

```
P(all clear) = 0.90 x 0.85 x 0.80 x 0.75
             = 0.459
```

Roughly **46%** — even though no single regulator is more likely than not to block it. **The intuition: clearance probabilities multiply across jurisdictions, so a deal that looks safe in any one country can be a coin flip globally; the binding constraint is the toughest reviewer, and adding jurisdictions only lowers the joint odds.** In reality the reviews are not fully independent (a structural problem visible to one agency is usually visible to all), but the multiplicative logic is why multinational deals carry wider arb spreads than domestic ones of similar concentration.

## Remedies: divestiture, behavioral conditions, or a block

When a deal has a real competitive problem but a fixable one, the agency does not simply block it — it negotiates a **remedy** that removes the harm while letting the rest of the deal proceed. Reading which remedy is likely is central to handicapping deal value, because a remedy *changes the value of the deal* without killing it.

![Decision flow showing how an agency clears a deal extracts a remedy or sues to block based on overlap and concentration](/imgs/blogs/antitrust-101-sherman-clayton-and-merger-review-5.png)

- A **divestiture** (a structural remedy) requires the merged firm to *sell* the overlapping business — a set of stores, a brand, a product line, a factory — to a credible independent buyer who will keep competing. This is the agencies' strongly preferred remedy because it restores competition cleanly and needs no ongoing policing. The cost to the parties is the lost revenue and profit of the divested business, plus the fact that they often sell it at a discount (a forced seller has weak negotiating power).
- A **behavioral remedy** (a conduct remedy) lets the firms keep all the assets but binds them with rules: a promise not to discriminate against rivals, to license a key input on fair terms, to keep a firewall between divisions. Behavioral remedies are cheaper for the parties but are disfavored by structure-focused enforcers, who distrust promises that require years of monitoring and are easy to evade. The 2023 posture pushed hard toward divestitures over behavioral fixes.
- A **block** is the outcome when no remedy can fix the harm — when the overlap is so central that carving it out would gut the deal's rationale, or when the agency simply does not believe a partial fix will work.

#### Worked example:

An acquirer agrees to buy a target for **\$10 billion**, expecting the combination to generate **\$1 billion a year** in combined operating profit. The agency finds an overlap in one product line and demands a divestiture of a business unit that contributes **\$120 million a year** of that profit. To win approval, the parties divest it — and because they are forced sellers on a deadline, they get only **\$700 million** for a unit that, at a normal **8×** profit multiple, might be worth 8 × \$120M = **\$960M**. The deal still closes, but the merged firm now earns \$1,000M − \$120M = **\$880M** a year instead of \$1 billion, and it took a roughly **\$260 million** haircut (\$960M fair value minus \$700M received) selling the unit under duress. **The intuition: a remedy is not free — it shaves both the recurring earnings and the one-time proceeds of the deal, and a trader handicapping a "remedy" outcome should value the merged entity *after* the carve-out, not at the full announced synergy.**

## Common misconceptions

Antitrust is full of intuitions that sound right and are wrong. Three misconceptions cost investors money repeatedly.

### Misconception 1: "Big means illegal."

This is the most common error, and we have already met its correction: **monopoly itself is legal; only the conduct or the merger that creates or abuses it is illegal.** A company with 90% market share that won that share by being better has broken no law. The Sherman Act Section 2 case against Google was not "Google is too big" — it was "Google used illegal *conduct* (default-payment exclusivity) to *maintain* its monopoly." The number that matters is not the market share itself but *how* it was won or kept, and whether a *merger* would increase it. Investors who panic-sell a dominant franchise on a "monopoly" headline, or who assume any large acquirer's deal is doomed, are pricing the wrong thing. The relevant question is always conduct and the specific deal's ΔHHI — not bigness.

### Misconception 2: "The agencies block most deals."

They do not — they clear the overwhelming majority. Of the thousands of HSR filings each year (often **3,000+** in an active year), only a low-single-digit percentage draw a second request, and only a small fraction of *those* end in a challenge. In a typical year the agencies file on the order of **20-40** merger challenges out of thousands of reviewable deals — well under **1-3%**. The base rate for a randomly chosen announced deal closing is very high. This matters enormously for merger arb: if you treat every deal as a coin flip, you will leave huge returns on the table on the deals that were always going to clear, and you will misprice the genuinely contested ones. The skill is *identifying the rare deal that is actually at risk* — high combined share, narrow market, an aggressive agency, overlapping products — not assuming risk everywhere.

### Misconception 3: "Only price matters."

For decades, under the consumer-welfare standard, the agency essentially asked one question: will this raise prices to consumers? If not, the deal was fine. That is no longer a safe assumption. The 2023 Guidelines and the New Brandeisian shift brought *non-price* harms squarely into the analysis: effects on **workers** (a merger that concentrates an employer's local labor market — monopsony — even if consumer prices are untouched), on **innovation** (a dominant firm buying a nascent competitor to kill a future product — the "killer acquisition" theory), and on **potential competition** (eliminating a firm that *would have* entered). A deal that looks clean on a pure price analysis — say, an acquisition where the target charges *consumers* nothing today — can still be challenged on innovation or labor grounds. The standard is evolving, and pricing a deal as if "no price hike means clearance" is a 2010s mental model applied to a 2020s regime.

## How it shows up in real markets

Antitrust is not abstract. It moves billions of dollars of deal value and franchise multiples in concrete, datable episodes.

**A deal break on antitrust.** When the agency sues and the parties walk, the target's stock falls back toward standalone value, often violently, in a single session. Acquirers also pay **break fees** (and sometimes *reverse* break fees the acquirer owes the target if regulators block the deal) running into the hundreds of millions or billions of dollars — a real cash transfer that itself prices into both stocks. A merger-arb book that was long the target into a block takes the full standalone-to-deal-price gap as a loss; this is the tail that defines the strategy's risk. The recurring lesson from blocked deals is that the warning was usually visible months earlier in the *concentration math* and the *second request* — the market that read those signals avoided the loss.

**An HHI-driven divestiture.** The most common visible outcome on a contested-but-fixable deal is a *consent decree with divestitures*. Grocery mergers, supermarket-pharmacy combinations, and beverage deals routinely close only after the parties agree to sell dozens or hundreds of overlapping stores or a portfolio of brands to a rival. The stock reaction is usually positive-but-muted: the deal clears (good), but at a smaller scale than announced (the carve-out shaves the synergy). Reading the *likely size* of the required divestiture from the overlap map is exactly the work that separates a good deal handicapper from a bad one.

**A multi-year review.** Some deals — especially large platform and vertical combinations — sit in review for *years*, the spread grinding wide as the second request drags and litigation looms. For merger arb, time is the silent killer: a deal that clears after two years instead of six months may still pay out at the deal price, but the annualized return collapses and the capital was tied up far longer than modeled. The biggest, most strategically important deals are often the ones that take longest, because they are exactly the ones the agencies fight hardest.

**A monopolization ruling against a dominant franchise.** The rarest but most consequential episode is a Section 2 case against a dominant company, like the Google search ruling. The stock often *does not* move much on the liability finding itself, because the market spent years assigning a probability to it — that is the discounting mechanism at work. The real repricing risk sits in the *remedies* phase: a behavioral remedy (change the default contracts) is a slap on the wrist that the market shrugs off, while a *structural* remedy (divest a browser, an ad exchange, share the underlying data) can permanently lower the franchise's growth rate and compress its multiple. This is the antitrust version of an *r*-shock versus a *g*-shock: a fine or a conduct tweak is a one-time cost the market absorbs, but a forced divestiture or a capped business model is a permanent haircut to the cash-flow stream. Handicapping a dominant-franchise antitrust case is therefore mostly about handicapping the *remedy*, not the liability finding.

The defense-sector repricing after geopolitical shocks, the foundry concentration we computed above, and big-tech multiple compression under platform-regulation risk are all the same mechanism: a rule (or a rule-regime shift) changing the value of a deal or a franchise *before* the rule fully bites. The market discounts the expected outcome continuously, which is why the news that *moves* a stock is rarely the ruling itself — it is the surprise relative to what was already priced.

## The playbook: how to handicap a deal

Antitrust analysis pays off when it tells you what to *do* — which deals to trust, which to fear, and what to watch. Here is the practitioner's checklist for turning the law into a position.

**1. Read the concentration math first.** Before anything else, define the narrowest plausible market and compute the combined share and the ΔHHI (`2 × share_A × share_B` for a horizontal deal). A deal that produces a post-merger HHI above 1,800 with a ΔHHI above 100 — or a combined share above 30% in a tightly drawn market — is presumptively problematic and should carry a *wide* spread. A deal between firms with single-digit shares in a fragmented market is, statistically, a near-certain clear. Most of your edge comes from *not* treating these two cases the same.

**2. Treat the second request as the loudest signal.** The single most informative event in a deal's life (short of an actual lawsuit) is the issuance of a second request. It re-rates the timeline (add 6-18 months) and signals serious investigation. If you are long a target into a second request, you are now in a fundamentally different, riskier trade than you were the day before, and the spread should — and usually does — widen sharply to reflect it. Conversely, *early termination* of the waiting period is a strong all-clear.

**3. Estimate the remedy-vs-block odds from the overlap.** When a deal is contested, the key question is whether the harm is *carve-out-able*. If the overlap is in one product line or one geography, the likely outcome is a divestiture and a (smaller) close — value the merged entity *after* the carve-out. If the overlap is central to the deal's whole rationale, or the harm is a diffuse platform/innovation theory that no divestiture cleanly fixes, the risk skews toward a block. The agencies' published *preference for structural over behavioral remedies* tells you a behavioral fix offer is a weaker signal than a clean divestiture offer.

**4. Back out the implied probability of approval.** Use the merger-arb spread. With a deal price `D`, a standalone fallback price `S`, and a market price `P`, the rough implied probability of close is `p ≈ (P − S) / (D − S)` (ignoring the discount for time). If a \$100 deal with an \$80 standalone trades at \$96, the implied close probability is `(96 − 80) / (100 − 80) = 80%`. Compare *your* estimate of approval odds — built from the HHI, the market definition fight, the agency's posture, and the second-request status — against the market's implied probability. Trade the gap. This is the bridge to the full [merger-arb position-sizing](/blog/trading/law-and-geopolitics/merger-arbitrage-trading-regulatory-deal-risk) framework.

#### Worked example:

You are handicapping a contested horizontal deal. Your scenario analysis: **60%** chance of a clean clearance at the **\$100** deal price; **25%** chance of a clearance *with a divestiture* that, by trimming the combined entity, leaves the target worth **\$96**; and **15%** chance of a *block*, sending the target back to its **\$80** standalone. The probability-weighted value is:

```
expected value = 0.60 x $100 + 0.25 x $96 + 0.15 x $80
               = $60.00 + $24.00 + $12.00
               = $96.00
```

If the target currently trades at **\$93**, the market is implying a *lower* probability of a good outcome than your analysis — your model says \$96, so the \$3 gap is your edge (before financing and time costs). If it trades at **\$98**, the market is more optimistic than you are, and the risk-reward is poor: you are being paid almost nothing to bear a 15% chance of a \$16 drop to \$80. **The intuition: build the antitrust outcome tree — clear / remedy / block — assign probabilities from the concentration math and the procedural signals, weight the values, and compare to the price; the whole discipline of trading deal risk reduces to this one calculation done honestly.**

![Bar chart of probability weighted deal value across clear remedy and block outcomes with the weighted value at ninety six dollars](/imgs/blogs/antitrust-101-sherman-clayton-and-merger-review-7.png)

**5. Know what invalidates the view.** The thesis that a deal clears is invalidated by: a second request (re-rate the timeline immediately); a public statement of concern from the agency head; a narrow market definition winning in the agency's complaint; the *withdrawal and refiling* of the HSR (a tactic that resets the clock and signals the parties are buying time to negotiate); and the appearance of a *competing* theory of harm (labor, innovation) that no divestiture fixes. The thesis that a deal *breaks* is invalidated by a remedy offer the agency signals it will accept, a court loss for the government on a similar theory, or a change in agency leadership toward a more permissive posture. The durability of a *dominant franchise* (the Section 2 question) is invalidated by a monopolization ruling with structural remedies — a forced breakup or divestiture is the tail that re-rates the whole multiple.

The deeper point ties back to the series spine: a statute written in 1890 and 1914, interpreted through guidelines revised in 2023, decides which deals happen and which monopolies stand. The law changes the rules of the game; the market discounts the expected outcome into the target's spread and the dominant firm's multiple *before* any court rules; the practitioner reads the concentration math and the procedural signals early, sizes the repricing, and knows what invalidates the view. That is antitrust as a market force.

## Further reading & cross-links

Within this series:

- [Merger arbitrage and trading regulatory deal risk](/blog/trading/law-and-geopolitics/merger-arbitrage-trading-regulatory-deal-risk) — the practitioner's deep dive: sizing a merger-arb position from the spread, the implied probability of approval, and the full clear/remedy/block tree.
- [Big tech antitrust and the New Brandeisians](/blog/trading/law-and-geopolitics/big-tech-antitrust-the-new-brandeisians) — how the evolving standard prices platform-regulation risk into mega-cap multiples.
- [Regulatory risk as an asset-pricing factor](/blog/trading/law-and-geopolitics/regulatory-risk-as-an-asset-pricing-factor) — why rule-exposed names trade at a discount, and how to quantify it.
- [How law moves markets: the transmission chain](/blog/trading/law-and-geopolitics/how-law-moves-markets-the-transmission-chain) — the series spine: law → policy → macro → prices → the trade.

Cross-asset and equity context:

- [Building a DCF, part 1: forecasting](/blog/trading/equity-research/building-a-dcf-part-1-forecasting) and [multiples 101: P/E, EV/EBITDA, P/B, P/S, PEG](/blog/trading/equity-research/multiples-101-pe-ev-ebitda-pb-ps-peg) — how a divestiture or a blocked deal flows into valuation.
- [Economic moats: durable competitive advantage](/blog/trading/equity-research/economic-moats-durable-competitive-advantage) — what antitrust risk does to the durability of a moat.
- [Capital allocation: the CEO's most important job](/blog/trading/equity-research/capital-allocation-the-ceos-most-important-job) — why a blocked deal forces a capital-allocation rethink.
- [Risk management as a business function](/blog/trading/hedge-funds/risk-management-as-a-business-function) — sizing event-driven deal risk in a portfolio context.
