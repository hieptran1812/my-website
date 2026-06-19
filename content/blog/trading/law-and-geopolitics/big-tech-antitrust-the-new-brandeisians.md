---
title: "Big-Tech antitrust: the new Brandeisians and the platform-regulation trade"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Why the antitrust war on the largest technology platforms is the biggest regulatory overhang on the index, and how a fine, a behavioral remedy, or a breakup reprices very different amounts of value."
tags: ["antitrust", "regulation", "big-tech", "digital-markets-act", "monopoly", "platform-regulation", "index-concentration", "regulatory-risk", "equity-research", "geopolitics", "valuation", "remedies"]
category: "trading"
subcategory: "Law & Geopolitics"
author: "Hiep Tran"
featured: true
readTime: 44
---

> [!important]
> **TL;DR** — A wave of antitrust action against the largest technology platforms — DOJ and FTC cases in the US, the European Union's Digital Markets Act — is the single biggest regulatory overhang on the mega-cap stocks that dominate the index, and *how* each case resolves reprices a huge share of global equity-market value.
>
> - The remedy type is the whole trade. A **fine** is a one-off cash hit (often well under 1% of market cap); a **behavioral remedy** dents a specific revenue line *forever*; a **breakup** re-rates the parts. These differ by an order of magnitude in dollar impact.
> - The biggest risk is almost never the fine. A several-billion-dollar fine on a trillion-dollar company is a rounding error; a rule that caps a 30% app-store take rate or kills a \$20B default-payment is a permanent margin event that compresses the multiple.
> - Because the top handful of names are roughly a third of the S&P 500, platform-regulation risk is not a single-stock story — it is an **index-level factor**. A remedy that reprices the mega-caps reprices the whole market.
> - The one number to remember: a permanent **30%→15%** cut to a platform's take rate, fed into a 25x-earnings stock, is worth roughly a **40%+ decline in that segment's value** — vastly more than any fine the same case could produce.

On August 5, 2024, a US federal judge ruled that Google had illegally maintained a monopoly in online search. The decision in *United States v. Google LLC* ran to 277 pages and concluded, in the now widely-quoted line, that "Google is a monopolist, and it has acted as one to maintain its monopoly." It was the most consequential American monopoly ruling since the Microsoft case a generation earlier. Alphabet's stock barely flinched that day — it had been a brutal session for the whole market on an unrelated yen-carry unwind — but the ruling set in motion a remedies fight whose outcome could reshape how the world's most-used product makes money, and could simultaneously cost Apple one of its most profitable, lowest-effort revenue streams.

That last point is the tell. A single antitrust ruling against *one* company put a question mark over a high-margin revenue line at a *completely different* trillion-dollar company. That is what makes Big-Tech antitrust different from a normal regulatory headline. These platforms are wired into each other and into the index. The cases are not about whether a firm pays a fine. They are about whether the *business model* survives in its current form — and the business models in question generate a meaningful fraction of the entire equity market's earnings.

This post builds the whole picture from zero: why platforms are a genuine puzzle for hundred-year-old antitrust law, what the "new Brandeisian" movement actually argues, what the live cases are, what the EU's Digital Markets Act already changed, and — the part that pays — how each kind of remedy maps to dollars of repriced value. We stay strictly analytical: the goal is not to argue whether breaking up Big Tech is good policy, but to teach you how to price the outcomes so you can read the overhang and size the trade. For the general machinery of how a legal threat becomes a price, see [how law moves markets through the transmission chain](/blog/trading/law-and-geopolitics/how-law-moves-markets-the-transmission-chain) and [regulatory risk as an asset-pricing factor](/blog/trading/law-and-geopolitics/regulatory-risk-as-an-asset-pricing-factor); here we apply that machinery to the single most valuable corner of the market.

![Flow from an antitrust case to fine behavioral remedy or breakup and how each maps to a new equity price](/imgs/blogs/big-tech-antitrust-the-new-brandeisians-1.png)

## Foundations: why platforms break the old antitrust playbook

Antitrust law — the law against monopolies and anticompetitive conduct — was written for a physical economy of railroads, steel, oil, and sugar. To trade the modern cases, you have to understand why digital platforms strain every concept that law was built on. Let us build each idea from the ground up.

### What antitrust law is, in one paragraph

In the US, two statutes do most of the work. The **Sherman Act of 1890** outlaws (Section 1) agreements that restrain trade and (Section 2) *monopolization* — not being big, but using improper means to obtain or maintain monopoly power. The **Clayton Act of 1914** added rules on mergers and specific practices. Two agencies enforce these: the **Department of Justice (DOJ)** Antitrust Division and the **Federal Trade Commission (FTC)**. The deep mechanics of these statutes and how merger review works are covered in [antitrust 101: Sherman, Clayton, and merger review](/blog/trading/law-and-geopolitics/antitrust-101-sherman-clayton-and-merger-review); here we need only the headline: monopolization cases ask whether a dominant firm crossed the line from *competing hard* to *excluding rivals improperly*.

For roughly forty years, US courts judged that line using the **consumer-welfare standard**: conduct is illegal if it harms consumers, and the clearest proof of harm is *higher prices or lower output*. If a firm got big but prices fell and output rose, the standard generally gave it a pass. This standard, associated with the Chicago School and championed in Robert Bork's 1978 book *The Antitrust Paradox*, became the dominant frame after the late 1970s. It worked tolerably for a world where "monopoly" meant a firm jacking up the price of a physical good.

It helps to be precise about what a US monopolization case actually has to prove, because the trade hinges on the *probability* a plaintiff clears that bar. Under Section 2 of the Sherman Act there are two elements: (1) the defendant possesses **monopoly power** in a properly defined market, and (2) it acquired or maintained that power through **exclusionary conduct** rather than "a superior product, business acumen, or historic accident." The first element turns on **market definition** — and market definition is where half the platform cases are won or lost. If the relevant market is "general search engines," Google's share is overwhelming; if it is "all ways a consumer might find information online" (including Amazon for products, social apps, AI chatbots), the share looks far smaller. The plaintiff wants a narrow market; the defendant wants a broad one. The second element asks whether the conduct *excluded* rivals improperly. Paying to be the default, bundling, self-preferencing, and acquiring nascent competitors are the recurring fact patterns. Critically, *having* a monopoly is legal in the US; *maintaining* it through exclusion is not. That distinction — lawful dominance versus unlawful exclusion — is the fulcrum every case turns on, and it is genuinely uncertain how a court will rule, which is exactly why these cases carry a *risk premium* rather than a known outcome.

### Why platforms confuse the price-based test

Now apply that test to Google Search, Facebook, or Amazon's marketplace, and watch it short-circuit on four features.

**Zero-price services.** Search and social media are *free* to the user. If the test for harm is "did the price go up?", a service whose price is permanently zero looks harmless by construction. You cannot show a price increase on a product that has no price. The harm, if any, shows up elsewhere — in the *quality* of the service, in the *privacy* surrendered, in the *advertisers* who pay, or in the *rivals* foreclosed — none of which the simple price test measures well.

**Network effects.** A platform gets *more useful* the more people use it. A social network with all your friends beats an empty one; a marketplace with every seller beats a sparse one; a search engine with the most queries learns the fastest. Network effects create a powerful tendency toward **winner-take-most** outcomes: scale begets scale. The hard question is whether a dominant position won by genuine network effects is a *deserved* prize for building a better product, or a *moat* that improperly locks out anyone who might build something better. Antitrust law has no clean answer.

**Two-sided (multi-sided) markets.** A platform serves *two distinct groups at once* and mediates between them: search serves users and advertisers; a marketplace serves shoppers and merchants; an app store serves consumers and developers. The platform often *subsidizes* one side (free search, free shipping perks) and *monetizes* the other (advertisers, app-store fees). This makes "price" almost meaningless: which price — the zero one users pay, or the rising one the other side pays? The US Supreme Court grappled with exactly this in *Ohio v. American Express* (2018) and made two-sided market cases harder for plaintiffs by requiring them to show harm across the *whole* platform, not just one side.

**Data and lock-in.** Platforms accumulate data that improves the product and raises the cost of switching. Your photos, your purchase history, your social graph, your saved settings — all of it makes leaving painful. Economists call this a **switching cost**, and high switching costs let an incumbent keep customers even when a rival is better. None of this shows up as a "price."

![Matrix of Big Tech antitrust cases by agency theory revenue at risk and remedy in play](/imgs/blogs/big-tech-antitrust-the-new-brandeisians-2.png)

The figure above is the map you should carry through the rest of this post: every live case pairs a *company* with an *agency*, a *legal theory*, a specific *revenue line at risk*, and a *remedy in play*. The trade is always about the third and fourth columns — the revenue line and the remedy — never about the firm in the abstract.

### The new Brandeisians: bigness as the harm

Into this confusion stepped a movement that rejects the price-based test outright. Named after Justice **Louis Brandeis** (1856–1941), who warned a century ago about the "curse of bigness" and the political danger of concentrated economic power, the **new Brandeisians** argue that antitrust lost its way when it narrowed to consumer prices. Their core claim: bigness itself — concentrated control over essential digital infrastructure — is a harm worth policing, even when prices are low or zero, because it stifles innovation, gives a few firms gatekeeper power over commerce and speech, and concentrates wealth and political influence.

The intellectual spark is often dated to a 2017 law-review note, *Amazon's Antitrust Paradox* by Lina Khan, which argued that Amazon could be a monopoly problem even though it relentlessly *lowered* prices, because the price-only lens missed the structural power it was accumulating. Khan later became Chair of the FTC (2021–2025), and the movement's thinking shaped a more aggressive enforcement posture at both the FTC and the DOJ during that period — including the cases below. Whether you find the movement persuasive is a policy question outside our scope. What matters for the trade is that **the enforcement standard itself became contested and more aggressive**, which raises the *probability* of an adverse outcome in any given case — and probability times severity is the whole of regulatory risk.

A practical caution that runs through this post: enforcement posture is not permanent. Agency leadership changes with administrations, and courts — not agencies — write the final word. A more aggressive FTC raises the *odds* a case is brought; it does not guarantee the *outcome*, which can take years and survive multiple changes of government. Treat the posture as a thumb on the probability scale, not as a verdict.

## The live cases: company by company

Here are the cases that matter, each mapped to the revenue line it threatens. These are public filings and rulings; specifics evolve, so treat the dollar figures as order-of-magnitude anchors for the *mechanism*, not as precise current values.

### DOJ v. Google — search (the 2024 monopoly ruling)

The DOJ, joined by dozens of states, sued Google in October 2020 alleging it illegally maintained a monopoly in general search. The core conduct: Google paid enormous sums — reported in the range of tens of billions of dollars a year — to be the *default* search engine on devices and browsers, most prominently on Apple's iPhone (Safari) and on Android. The theory: paying for default placement foreclosed rivals from the scale they would need to compete, because most users never change a default.

In August 2024 the court agreed that Google held monopoly power in search and search advertising and had maintained it unlawfully through those exclusive default deals. The case then moved to the **remedies phase** — the separate proceeding that decides *what to do about it*. Possible remedies range from banning the default-payment deals (a behavioral remedy) to, at the aggressive end, forcing Google to divest parts of its business such as the Chrome browser (a structural remedy). The revenue line at risk: the search-traffic moat itself, and — crucially — the payments flowing *to Apple*, which we treat as its own linked risk below.

Why does the default matter so much? Because of a behavioral fact the court leaned on heavily: **most users never change a default**. The search box that ships pre-selected captures the overwhelming majority of queries, and queries are the raw material that trains the engine and attracts advertisers. By paying to *own* the default slot across the most valuable surfaces — the iPhone's browser above all — Google ensured rivals could never reach the query scale they would need to improve and compete. The plaintiffs' theory is that this is exclusion, not merit: rivals lost not because their product was worse but because the on-ramp was bought. The defense is that users *choose* Google because it is better, and switching a default takes seconds. A court found for the plaintiffs on liability; the remedies fight is where the dollars are decided.

### DOJ v. Google — ad tech

A separate DOJ case, filed in 2023, targets Google's **advertising-technology** stack — the plumbing that runs the auctions placing display ads across the web. The allegation is that Google owns the dominant tool on *every* side of the ad auction (the seller's ad server, the buyer's tools, and the exchange in the middle) and uses that to self-deal. In 2025 a court found Google liable for illegally monopolizing parts of the ad-tech market, opening a remedies fight that could force Google to *divest* ad-tech assets — a structural remedy aimed squarely at a revenue line.

The ad-tech case is the textbook *vertical-integration* problem dressed in digital clothes. Picture an auction where the same firm owns the auctioneer, runs the tool the buyers use to bid, and runs the tool the sellers use to list — and also bids in the auction itself. Even if every individual step looks efficient, owning all three sides creates the *opportunity and incentive* to set the rules in your own favor: route demand to your own exchange, take a clip at each layer, and starve rival exchanges of liquidity. The DOJ's theory is that Google did exactly this. The remedy in play is structural — a forced divestiture of the ad-server or exchange business — which is why this case, unlike the search case, has a *breakup* squarely on the table for a specific, separable revenue line rather than for the whole company. For the investor, an ad-tech divestiture is a cleaner sum-of-the-parts exercise than a full-company breakup, because the unit being carved out is discrete and the rest of the business is untouched.

### FTC v. Amazon

The FTC, with state attorneys general, sued Amazon in September 2023, alleging it uses an illegal monopoly to inflate prices and degrade quality. The theory leans new-Brandeisian: Amazon allegedly punishes sellers who offer lower prices elsewhere and effectively coerces them into using its fulfillment and advertising services to stay visible. The revenue lines at risk are the marketplace **seller fees** and the fast-growing **advertising** business that sits on top of the marketplace. A possible structural remedy would separate the marketplace from the logistics/fulfillment arm.

The Amazon case is the purest test of the new-Brandeisian thesis, because Amazon's defining trait is *low prices* — the very thing the old consumer-welfare standard treats as proof of innocence. The FTC's answer is that the harm is structural and shows up not as a sticker-price increase but as *fees extracted from sellers* (who pass them on) and as *degraded quality* (a marketplace stuffed with paid placements). Watch this case closely as a *standard-setting* event: if a court accepts a theory of harm that does not require showing higher consumer prices, it lowers the bar for every other platform case and raises the probability term in everyone's regulatory-risk calculation. The single most valuable revenue line here is the advertising business — a high-margin stream that has grown into one of the largest ad platforms on earth and that the market increasingly values like a separate, premium-multiple company embedded inside the retailer.

### FTC v. Meta

The FTC's case against Meta is the clearest *breakup* case on the board. It alleges that Meta (then Facebook) bought **Instagram** (2012) and **WhatsApp** (2014) to neutralize nascent competitors before they could threaten its social-networking dominance — the so-called "kill-zone" acquisition theory. The remedy the FTC seeks is structural: *unwind* the deals and spin Instagram and WhatsApp back out as independent companies. This is the case where a breakup is the explicit ask, not a tail scenario.

The Meta case is analytically interesting because it attacks *consummated* mergers — deals that regulators reviewed and cleared years ago. The FTC's challenge is partly that the social-networking market should be defined narrowly (excluding TikTok and YouTube as different categories), so that Meta still looks dominant despite obvious rivals for attention. If the market is defined broadly, Meta's "monopoly" looks shakier. As with every platform case, **market definition is doing the heavy lifting**, and the same uncertainty that makes the legal outcome hard to predict is what creates the tradeable overhang. For the investor, the key point is that a *forced divestiture* of Instagram is the explicit remedy on the table here — so the sum-of-the-parts math below is not hypothetical for this name, it is the actual scenario to value.

### Apple — the App Store and the DOJ case

Apple faces antitrust pressure on two fronts. First, the **App Store**: Apple takes a commission — historically up to **30%** — on App Store sales and in-app purchases, and restricts developers from steering users to cheaper payment options outside the app. This take rate has been challenged in courts (the *Epic v. Apple* litigation), squeezed by regulators, and is the single most-watched revenue-line risk in the group. Second, in 2024 the **DOJ sued Apple** for monopolizing the smartphone market through conduct that allegedly locks users in and locks rivals out (degrading cross-platform messaging, restricting third-party wallets and apps). The App Store commission and the broader iPhone lock-in are the revenue lines at risk.

## The EU's Digital Markets Act: the remedy written into law

While the US fights case by case in court, the European Union took a different route — and it is the more immediately consequential one for the business models, because it does not wait for a ruling.

The **Digital Markets Act (DMA)**, in force since 2023 and applied from 2024, designates the largest platforms as **"gatekeepers"** — firms that operate a "core platform service" (a search engine, an app store, an operating system, a social network, a browser) at a scale that controls access between businesses and consumers. Once designated, a gatekeeper must obey a list of *per-se* obligations — rules that apply automatically, with no need to prove harm in a courtroom. In other words, the DMA writes the behavioral remedy straight into statute and skips the multi-year liability fight.

![Tree of EU Digital Markets Act gatekeeper obligations and the concrete product change each forces](/imgs/blogs/big-tech-antitrust-the-new-brandeisians-6.png)

The headline obligations, each of which has already forced a concrete product change in Europe:

- **No self-preferencing**: a gatekeeper cannot rank its own products or services above rivals' in its own marketplace or search results.
- **Interoperability**: messaging services must open up to third parties; an operating system must let rival services plug in.
- **Sideloading / alternative app stores**: a mobile OS must allow apps and app stores from outside its own store — directly threatening the App Store commission model in Europe.
- **No tying**: a gatekeeper cannot force users to sign up for one service to use another, and must allow users to uninstall pre-loaded apps and change defaults easily.
- **Data access and portability**: business users must be able to access the data they generate and take it elsewhere.

The DMA's enforcement teeth are real: fines can reach **up to 10% of a gatekeeper's total worldwide annual turnover**, and **up to 20% for repeat infringements** — and, critically, the EU can impose *structural* remedies (including divestitures) for systematic non-compliance. That 10–20%-of-global-revenue cap is an order of magnitude larger than the fixed-euro fines of the old regime, and it is why the DMA changed product design and not just legal budgets. In 2024–25 the EU opened non-compliance investigations and levied DMA fines against several gatekeepers, forcing changes to app-store rules, default-browser choice screens, and self-preferencing in search.

The EU's older, case-by-case antitrust regime is still running too: the European Commission has fined Google a cumulative sum well above **€8 billion** across the Shopping (2017, €2.42B), Android (2018, €4.34B), and AdSense (2019, €1.49B) cases. Those fines, large as they sound, were dwarfed by the *repricing* the cases signaled — exactly the point of the next section, and the reason a fine is the least interesting outcome.

The DMA's revenue-based fine cap deserves a moment of its own, because it is the one place where even a *fine* gets genuinely large. The old fixed-euro fines were capped at a level that, for a trillion-dollar company, rounded to noise. The DMA flips that: 10% of *total worldwide annual turnover*, rising to 20% for repeat infringement, scales the penalty to the *size of the firm* rather than to the size of the infringement.

#### Worked example: a DMA fine at 10% of global turnover

Take a gatekeeper with **\$350 billion** of total annual revenue. A maximum DMA fine at 10% of worldwide turnover is:

\$350B × 10% = **\$35 billion** — and at the 20% repeat-infringement rate, **\$70 billion**.

Compare that with a pre-DMA fixed fine of a few billion: the DMA penalty is an order of magnitude larger, and large enough to matter even against a trillion-dollar market cap (a \$35B fine is ~2–3% of a \$1.2–1.5T cap, versus the ~0.3% of the old regime). *The intuition: most fines are noise you divide by market cap, but the DMA's turnover-scaled cap is the rare fine big enough to move the needle on its own — though even here, the behavioral obligations the fine enforces are the larger long-run risk to the business model.* The behavioral changes the DMA forces — sideloading, alternative payments, no self-preferencing — still dwarf the fine over time, but the fine is no longer a rounding error.

## The App Store and the 30% take rate: the single most-watched revenue line

Of all the revenue lines on the case map, the App Store commission deserves its own section, because it is the cleanest example in the group of a behavioral remedy hitting a near-pure-margin stream — and because it is under pressure from three directions at once.

The model is simple. Apple (and, with a lower headline rate, Google's Play store) controls the only way to install software on its mobile operating system, and takes a commission — historically **up to 30%**, with reduced **15%** tiers for small developers and for subscriptions after the first year — on paid apps and in-app purchases. It also enforced **anti-steering** rules that barred developers from even *telling* users that a cheaper subscription was available on the web. Because the store is already built, the marginal cost of processing one more transaction is close to zero, so this commission is **almost pure margin** — it drops straight into the high-multiple Services segment.

Three forces are squeezing it. First, **litigation**: the *Epic v. Apple* case (the *Fortnite* dispute) did not break the commission outright, but it forced Apple to relax anti-steering rules in the US, letting developers point users to outside payment options — a crack in the wall. Second, **the DMA**: in Europe, Apple must now permit alternative app stores and sideloading, and must let developers use and advertise third-party payment systems, which directly threatens the commission on the affected transactions. Third, **regulatory pressure elsewhere** (South Korea, Japan, and others) pushing similar steering and alternative-payment rules. Each crack does not eliminate the commission, but each one shifts some volume to cheaper rails and caps how high the effective take can stay.

The reason this revenue line is so closely watched is *leverage on the multiple*. The market pays a premium multiple for Services revenue precisely because it is high-margin, recurring, and growing faster than hardware. So a dollar of App Store profit at risk is worth *more* in market cap than a dollar of hardware profit — a take-rate cut hits both the *numerator* (less profit) and the *denominator* (a lower multiple, because the high-quality Services stream now carries regulatory risk). That double hit is why an App Store headline can move the stock far more than the immediate revenue math alone would suggest. The arithmetic of a take-rate cut is the worked example below; carry this section into it.

## How a remedy maps to dollars: the heart of the trade

This is the section that pays for the post. Three outcomes — fine, behavioral remedy, breakup — and they differ by orders of magnitude in dollar impact. Get this hierarchy right and you will never again confuse a scary headline (a big fine) with a real threat (a margin event).

Before the dollar math, fix the legal vocabulary, because the words map directly to magnitudes. A **fine** (or "disgorgement," or "penalty") is a one-time payment — a cash outflow you subtract once. A **behavioral remedy** (also called a *conduct* remedy) is a court order or statute that changes *how the firm operates* going forward: stop the exclusive default deals, end self-preferencing, allow alternative payment systems, publish interoperability interfaces. A **structural remedy** changes *what the firm is*: divest a business, spin off a unit, unwind a past acquisition — the breakup. Courts in the US generally prefer the *least intrusive* remedy that restores competition, which usually means they reach for behavioral remedies first and treat structural ones as the exceptional last resort — a meaningful fact for probability-weighting, because it tilts the distribution toward conduct remedies and away from breakups in most (not all) US cases. The EU, through the DMA, can impose behavioral rules by statute *without* a court fight and reserves structural remedies for systematic non-compliance. Knowing which remedy a given case is most likely to produce is the first input to sizing the trade.

![Tree of remedy scenarios fine behavioral remedy and breakup with their dollar impact](/imgs/blogs/big-tech-antitrust-the-new-brandeisians-3.png)

### A fine is a one-off cash outflow

A fine is the simplest and usually the *smallest* outcome. It is a one-time cash payment. To value its impact, you subtract it from the firm's value and stop. On a trillion-dollar company, even a multi-billion-dollar fine is a fraction of a percent. The market knows this, which is why the *stock reaction* to a fine is almost always larger than the fine itself — the reaction is pricing what the fine *signals* about future conduct rules, not the cash.

#### Worked example: a \$5B fine on a \$1.5T company

Suppose a regulator fines a platform \$5 billion. The company's market capitalization is \$1.5 trillion. The pure cash impact is:

\$5B / \$1,500B = **0.33% of market cap**.

If the firm trades at 25x earnings and earns \$60B a year, the fine is also about 8% of *one year's* profit — painful for a quarter, invisible over a decade. By contrast, suppose the same case also forces a conduct change that permanently shaves 3% off annual revenue and, through operating leverage, 5% off earnings. Capitalized at the same 25x, a permanent 5% earnings cut is worth roughly **5% of market cap = \$75 billion** — fifteen times the fine. *The intuition: a fine is a one-off you divide by market cap and forget; a behavioral remedy is a perpetuity you capitalize, and a small permanent dent in cash flow dwarfs a large one-time payment.*

### A behavioral remedy dents a revenue line — forever

A behavioral remedy changes how the firm is *allowed to operate*: stop self-preferencing, stop paying for defaults, let developers steer to outside payments, cap a fee. The key word is *forever*. Unlike a fine, a conduct rule reduces a revenue or margin stream into perpetuity, and perpetuities are expensive. You value the hit by estimating the annual cash-flow reduction and capitalizing it at the firm's multiple, then — second order — by the *multiple* the market is willing to pay, because a business newly exposed to ongoing conduct rules is a riskier, slower-growing business that earns a lower multiple.

This is where the App Store take rate sits, and it is the canonical example.

#### Worked example: cutting an App-Store-style take rate from 30% to 15%

Take a stylized platform that runs an app store. Annual gross sales flowing through the store (gross merchandise value, GMV) are **\$100 billion**. The platform takes a **30%** commission, so its app-store revenue is:

\$100B × 30% = **\$30B/year**.

This revenue is almost pure margin — the store is built, the marginal cost of processing another transaction is tiny — so assume a **90% operating margin**, giving \$27B of operating profit. Now a behavioral remedy (or a DMA-style rule, or competitive pressure from sideloading) forces the take rate down to **15%**:

New revenue = \$100B × 15% = **\$15B/year**, a \$15B revenue cut, and roughly **\$13.5B less operating profit**.

Capitalize that lost profit. If the segment was valued at the company multiple of, say, **25x operating profit**, the lost \$13.5B of profit is worth:

\$13.5B × 25 = **\$337.5 billion** of erased value.

Phrased as a percentage of the *segment*, halving the take rate roughly *halves* the segment's profit and therefore roughly halves its value — a 40–50% segment hit, depending on how much GMV migrates to cheaper rails versus stays. *The intuition: a take-rate cut is not a haircut on a fee, it is a permanent halving of a near-pure-margin profit stream, and you capitalize the whole thing — which is why it dwarfs any fine the same case could ever produce.*

### A breakup re-rates the parts

The most dramatic remedy is structural: split the firm. The reflexive assumption is that a breakup *destroys* value. Often it does the opposite. A sprawling conglomerate frequently trades at a **conglomerate discount** — the market pays *less* for the bundle than it would for the parts, because the bundle is hard to analyze, mixes high- and low-multiple businesses, and may cross-subsidize weak units. Splitting it lets each part be valued on its own — a high-growth unit earns a high multiple, a stable cash machine earns its own — and the sum of the parts can exceed the discounted whole.

![Before and after stacks comparing a conglomerate valuation with the sum of the parts after a breakup](/imgs/blogs/big-tech-antitrust-the-new-brandeisians-4.png)

#### Worked example: sum-of-the-parts vs the conglomerate

Suppose a platform conglomerate has three businesses, and as a single stock the market applies a **15% conglomerate discount** to the blended value because the mix is messy and the cloud unit drags on the reported multiple. Value the parts separately, at the multiple each *deserves*:

- Search + ads: \$70B profit × 18x = **\$1,260B**
- Cloud: \$15B profit × 14x = **\$210B**
- Video: \$18B profit × 24x = **\$432B**

Standalone sum of the parts = **\$1,902B**. But as one stock, the market applies the 15% discount:

\$1,902B × (1 − 0.15) = **\$1,617B** as a conglomerate.

Now split it. Freed from the discount, and with the cloud unit re-rated higher once investors can value it as a focused growth business (say 18x instead of 14x → \$270B) and the video unit getting a clean growth multiple, the post-breakup sum can land *above* \$1,902B. Even at the undiscounted \$1,902B, the breakup unlocks:

\$1,902B − \$1,617B = **\$285 billion** of value — about **18%** upside — simply by removing the discount.

*The intuition: a forced breakup is only value-destructive if the parts are worth less apart than together; for a diversified platform trading at a conglomerate discount, the split can re-rate the pieces and add value — which is why "breakup" is not automatically the bear case.*

The caveat — and it is a real one — is **lost synergy and scale**: shared infrastructure, cross-selling, a unified ad system, and a single data asset can make the integrated whole genuinely more valuable than the sum. Whether a breakup adds or destroys value is an *empirical* question about how much of the conglomerate's value is true synergy versus a discount on complexity. The trade is to estimate both and net them.

### The Google→Apple linkage: one remedy, two megacaps

The most elegant — and most under-appreciated — feature of the Google search case is that the remedy hits *two* companies. Google pays Apple to be the default search engine in Safari. That payment, reported in the range of **\$18–20 billion a year**, is almost pure profit to Apple: Apple does essentially nothing for it except keep a default setting. It flows straight to the bottom line in Apple's high-margin **Services** segment.

![Graph showing how a ban on default search payments hits both Google traffic and Apple high margin revenue](/imgs/blogs/big-tech-antitrust-the-new-brandeisians-5.png)

A behavioral remedy that bans Google from paying for default placement therefore lands on both names at once: Google loses its guaranteed traffic moat (rivals could win the default slot, or there is no paid default at all), *and* Apple loses a ~\$20B near-100%-margin payment it did almost nothing to earn.

#### Worked example: the Google→Apple default-payment hit to Apple

Assume Apple receives **\$20B/year** from Google for default placement, at essentially **100% margin** (it is found money). Apple trades at, say, **30x earnings** because its Services segment is high-margin and recurring and the market pays a premium for it. Capitalize the lost profit:

\$20B × 30 = **\$600 billion** of Apple market value tied to that single payment.

Now, the payment will not vanish entirely in every scenario — Apple might capture some search-monetization itself, or strike a non-exclusive arrangement — so haircut the loss to, say, half: a \$10B permanent profit hit × 30x = **\$300 billion**. Either way, the number is enormous, and it sits on a company that is not even a *defendant* in the search case. *The intuition: a single remedy in one company's case can erase hundreds of billions at a different company, because platform revenue lines are wired into each other — which is exactly why you map the linkage, not just the named defendant.*

This linkage is also why the cases are an *index* problem and not a single-name problem, which is the next idea.

## Index-concentration risk: why this is a systemic factor

Here is the structural fact that turns Big-Tech antitrust from a stock-picking question into a market-level one. The largest technology platforms are not just big companies — they are a *huge fraction of the index itself*. In the mid-2020s, the largest seven or so US technology and platform companies grew to roughly **a third of the S&P 500 by market capitalization** — a level of concentration not seen in decades. The exact weight moves daily and depends on which names you count, but the order of magnitude is the point.

![Diagram showing the top mega cap platforms as about a third of the S&P 500 making their legal risk systemic](/imgs/blogs/big-tech-antitrust-the-new-brandeisians-7.png)

When a small number of names dominate the index, *their* regulatory risk becomes *everyone's* risk. An investor who owns a broad index fund — which is most investors, including pension funds and retirement accounts — is implicitly making a concentrated bet on a handful of platform business models surviving in their current form. A remedy wave that compressed the multiple across these names would not just hurt a few stocks; it would drag the index, and with it the savings of anyone holding the market. This is why platform-regulation is best treated as a *factor* — a systematic exposure that touches a portfolio broadly — rather than as idiosyncratic single-stock news. The mechanics of treating a legal risk as a priced factor are developed in [regulatory risk as an asset-pricing factor](/blog/trading/law-and-geopolitics/regulatory-risk-as-an-asset-pricing-factor); the new wrinkle here is that the factor's exposure is *concentrated in the index's own top holdings*, so you cannot diversify it away by buying the market — buying the market *is* the bet.

There is a second-order layer that makes the concentration even more important. The exposed platforms are not just large *index weights* — they are also major *customers and suppliers* to each other and to the rest of the technology complex. The mega-caps buy each other's cloud capacity, advertise on each other's surfaces, and fund a large share of the capital spending that flows to chipmakers and equipment vendors. A remedy that forces one platform to retrench therefore ripples outward: less ad spending, less cloud demand, less capital expenditure. So the *direct* index-weight effect (a third of the S&P 500 reprices) is compounded by an *indirect* supply-chain effect (the names that sell into the platforms also wobble). The practical takeaway is that a naive "I'll just avoid the four defendants" hedge under-covers the exposure, because the risk leaks through the supply chain into names that are not even on the case map. The honest way to size it is at the *complex* level — the platforms plus their largest suppliers — not name by name.

#### Worked example: a fine versus the market-cap reaction

Recall the 2017 EU Google Shopping case: a **€2.42B** fine (about \$2.7B at the time) on a parent company worth roughly **\$650B**. The pure arithmetic says the stock should fall about:

\$2.7B / \$650B = **0.4%**.

The stock fell roughly **3%** intraday — about **\$19 billion** of market value — on a \$2.7B fine. The market move was about **seven times** the fine. Why? Because the market was not pricing the cash; it was raising the *probability* it assigned to a future business-model remedy and trimming the multiple it would pay for cash flows now exposed to a hostile regulator. Now scale that logic to the index: if the same multiple-trimming hit the seven mega-caps that make up a third of the S&P 500, even a modest 5% de-rating across those names would, by itself, knock roughly **1.7%** off the entire index (5% × 33%) — before any spillover to the names that *supply* the platforms. *The intuition: the fine is the headline you can compute on a napkin; the repricing is a multiple move on a perpetuity, and when it hits index-dominant names it becomes a market event, not a stock event.*

## The long timelines: why the overhang persists for years

A crucial feature for the trade: these cases move at the pace of the courts, not the news cycle. A monopoly case runs from complaint to trial to liability ruling to a *separate* remedies phase to appeals — and the appeals alone can take years and survive multiple administrations. The DOJ filed the Google search complaint in 2020; the liability ruling came in 2024; the remedies fight and the appeals stretch well beyond that.

![Timeline of an antitrust case from complaint to ruling to remedies to appeals showing the overhang lasts years](/imgs/blogs/big-tech-antitrust-the-new-brandeisians-8.png)

For the practitioner this cuts two ways. First, the **overhang is durable**: the regulatory discount sits on the stock for years, which means the *de-rating* can persist far longer than an event-trader expects, and there is no single "resolution day" to fade into. Second, the **repricing happens on rulings and signals, not on the distant effective date**: the stock moves the day a court finds liability, the day a remedy proposal leaks, the day an appeal succeeds or fails — long before any remedy actually bites. You trade the *information events* along the timeline, not the eventual remedy. This is the event-study logic of [how a rule becomes a price](/blog/trading/law-and-geopolitics/how-a-rule-becomes-a-price-expectations-drift-and-repricing), applied to a multi-year legal calendar.

The long timeline also creates a specific *behavioral* edge. Because the cases drag on for years, the market periodically *forgets* about them between catalysts, lets the regulatory discount narrow, and then re-prices abruptly when the next ruling lands. That oscillation — overhang fades into complacency, then snaps back on a ruling — is the rhythm an event-aware investor exploits: you want to *add* protection cheaply when complacency has compressed the implied volatility ahead of a known ruling window, not chase it after the gap. And because there are *multiple* cases against *multiple* names with *staggered* calendars, there is almost always a live catalyst somewhere in the complex, which is why platform-regulation behaves like a persistent factor rather than a one-off event. A further subtlety: a remedy can be *ordered* and then *stayed* pending appeal, so the cash-flow hit may be years away even after a "loss" — the market discounts the *expected* present value of that delayed, uncertain hit, which is smaller than the headline remedy and is exactly why the stock often falls *less* on a liability ruling than a naive reading of the remedy would imply.

## Common misconceptions

### Misconception 1: "The fine is the real risk"

This is the single most common error, and the worked examples above demolish it. A fine is a one-off you divide by market cap; a behavioral remedy is a perpetuity you capitalize. We saw a \$5B fine cost 0.33% of cap while a permanent 5% earnings dent cost 5% — *fifteen times more*. The EU has fined Google over **€8 billion** cumulatively, and Alphabet's market cap grew by *trillions* over the same span: the fines were noise; the business model kept compounding. **Watch the conduct rule, not the cash penalty.** A headline that screams about a record fine but buries a quiet change to how the firm can operate has the magnitudes exactly backward.

### Misconception 2: "A breakup destroys value"

Sometimes, but far from always. As the sum-of-the-parts example showed, a conglomerate trading at a 15% complexity discount can *gain* roughly 18% of its value from a split that lets each part earn its proper multiple. The classic historical analogue is the 1984 AT&T breakup: the "Baby Bells" plus the parent were, within a few years, worth substantially more than the pre-breakup whole. A breakup destroys value only when *true synergy* — shared data, shared infrastructure, cross-selling — exceeds the conglomerate discount. That is an empirical question to estimate, not an assumption to make. **The reflexive "breakups are bad for the stock" is often wrong; sometimes the breakup is the bull case.**

### Misconception 3: "It's already priced in"

The most dangerous phrase in markets. "Priced in" is not binary — it is a *probability-weighted average* of outcomes, and as that probability distribution shifts on each ruling, the price moves. Before the August 2024 liability ruling, the market priced *some* chance Google would win outright; the ruling removed that tail and shifted weight toward adverse remedies, which is information even if "antitrust risk" was broadly known. The same is true on the upside: a *favorable* appeal, a *settlement* that caps the damage, or a *behavioral* remedy that is milder than feared all re-rate the stock *up* because the bad tail gets truncated. **As long as the outcome is uncertain and the case is live, new rulings carry information — the overhang is never fully "priced in" until the distribution collapses to a single known outcome.**

### Misconception 4: "US and EU outcomes are the same trade"

They are not, and the divergence is itself tradeable. The EU's DMA already forced product changes in Europe — choice screens, sideloading, interoperability — *years* before the slow US court cases will produce remedies. This creates a **geographic split in the business model**: a platform may run one set of rules in Europe and another in the US, and the revenue impact differs by region. An investor must map the *European* revenue exposure to the DMA (already biting) separately from the *US* exposure to the court cases (a slower, more uncertain overhang). **Treating "Big-Tech regulation" as one undifferentiated risk misses that the EU is already a behavioral-remedy reality while the US is still a probabilistic court fight.**

## How it shows up in real markets

### The ruling-day move

The cleanest signal is the price reaction to a liability ruling or a remedies proposal. When a court finds a platform liable, or when a remedy proposal leaks as harsher or milder than expected, the stock gaps and the implied volatility that built up into the event collapses. Because these are *known-date-uncertain-outcome* events (the ruling date is often public; the content is not), they behave like binary catalysts — the options market prices a straddle into them, and the move on the day reflects the *surprise* relative to what was priced. For the general mechanics of trading a known-date legal catalyst, see [the regulatory calendar: trading the rulemaking clock](/blog/trading/law-and-geopolitics/the-regulatory-calendar-trading-the-rulemaking-clock).

### The App-Store fee-cut risk to Services revenue

A specific, watchable channel: any erosion of the App Store commission — from a court order, a DMA rule, or competitive sideloading pressure — flows directly into the high-margin **Services** revenue that the market pays a premium multiple for. The market reaction to App Store news is amplified precisely because Services carries a *higher* multiple than hardware: a dollar of Services profit at risk is worth more in market cap than a dollar of hardware profit. Watch the Services growth rate and the disclosed or estimated take rate; a step-down is a multiple event.

### The EU-vs-US product divergence

Watch for the same product shipping with *different rules* in Europe versus the US — alternative app stores live in the EU but not the US, choice screens in the EU only, interoperability features rolled out under DMA pressure. The European version is a live preview of what a US behavioral remedy might eventually look like, and the *European* revenue impact (already happening) is a leading indicator for how much a US remedy would bite. The ad-tech model risk and the parallel privacy-law squeeze are developed in [data and privacy law: GDPR and the tech business model](/blog/trading/law-and-geopolitics/data-and-privacy-law-gdpr-and-the-tech-business-model).

## The playbook: how to trade platform regulation

Here is the concrete, repeatable process. Every step maps a legal fact to a dollar of value and to a position.

**1. Map each case to the revenue line at risk.** Do not think "Apple has antitrust risk." Think "the App Store commission — call it \$X billion of near-pure-margin Services profit — is exposed to a take-rate cut, and the DOJ smartphone case threatens the lock-in." Use the case map: company × agency × theory × revenue line × remedy. The line, not the firm, is what reprices.

**2. Value the remedy scenarios, weighted by probability.** Build the simple decision tree: probability of fine × (small \$ hit) + probability of behavioral remedy × (capitalized revenue-line hit) + probability of breakup × (sum-of-the-parts re-rate, which may be *positive*). The expected value of the overhang is the probability-weighted sum; the *risk* is the dispersion across branches. A name where the bad branch (a take-rate cut) is severe and the probability is rising is a name to underweight or hedge.

#### Worked example: probability-weighting the remedy tree

Suppose for one platform you judge: 50% chance the case ends in a fine (≈ −0.5% of cap), 40% chance of a behavioral remedy (≈ −15% of the exposed segment, and the segment is 40% of the firm, so ≈ −6% of cap), and 10% chance of a breakup that you estimate is roughly value-neutral (≈ 0%). The probability-weighted impact is:

0.50 × (−0.5%) + 0.40 × (−6%) + 0.10 × (0%) = −0.25% − 2.4% + 0% = **−2.65% of cap**.

That −2.65% is the *expected* drag the overhang justifies. If the stock has de-rated *more* than that relative to peers without antitrust exposure, the overhang may be over-discounted (a long opportunity if a benign branch hits); if *less*, it is under-discounted (the de-rating has further to run if the behavioral branch firms up). *The intuition: the trade is the gap between the market's implied discount and your probability-weighted estimate — you are not predicting the verdict, you are pricing the distribution better than the consensus.*

**3. Treat platform-regulation as an index-level factor.** Because the exposed names are a third of the index, size the risk at the *portfolio* level, not just per-name. If you run a broad-market book, you already own this risk whether you chose to or not. Decide deliberately how much platform-regulation beta you want.

**4. Hedge the concentration.** If you want index exposure but not the platform-regulation tail, the toolkit is the standard one for a concentrated factor: underweight the most-exposed names relative to the index, pair a long index against short single-name exposure to the highest-risk defendant, or buy downside protection (puts) on the names with the nearest, most binary catalysts. Equal-weight index exposure mechanically dilutes the mega-cap concentration. The general mechanics of sizing and hedging a concentrated factor are in the [cross-asset](/blog/trading/cross-asset) and [equity-research](/blog/trading/equity-research) series; the point here is that the hedge target is *specific revenue lines at risk*, so a basket weighted by App-Store/default-payment/ad-tech exposure hedges the factor better than a generic tech short.

**5. Trade the timeline, not the headline.** The repricing happens on rulings, leaked remedy proposals, and appeal outcomes — not on the distant effective date. Build a calendar of the known catalyst dates (trial dates, expected ruling windows, appeal schedules, DMA compliance deadlines) and position into the ones where the implied move underprices the dispersion of outcomes.

**6. Know what invalidates the view.** A platform-regulation *bearish* thesis is invalidated by: a favorable appeal that overturns liability; a settlement that caps the remedy to a fine plus minor conduct tweaks; a change in enforcement posture (new agency leadership that drops or narrows a case); or evidence that a behavioral remedy *failed to dent revenue* (e.g., users stay on the default even when offered a choice screen — a real DMA finding in some markets, where most users do not switch). A *bullish* (overhang-overdone) thesis is invalidated by a harsher-than-expected remedy proposal or a second agency opening a new front. Write the invalidation down *before* you put on the trade, so a ruling that goes against you triggers a review, not a rationalization.

A note on the *behavioral-remedy-fails* invalidation, because it is the most under-appreciated one. The entire bear case for a conduct remedy assumes that *changing the rule changes user behavior* — that if you offer a choice screen, users will switch; if you allow sideloading, developers will leave the official store; if you ban the default payment, traffic will flow to rivals. In practice, defaults are extraordinarily sticky, brand and convenience are powerful, and early DMA evidence in some markets showed that even when users were *offered* a choice, the large majority stuck with the incumbent. If the remedy is imposed and revenue *barely moves*, the de-rating reverses — the market had priced a revenue hit that did not materialize. This is the single best example of why "the remedy" and "the revenue impact" are different variables: a court can change the rule and the business can keep most of the cash flow anyway. The disciplined investor models the remedy *and* the elasticity of user behavior to it, and treats a low elasticity as the bull case hiding inside a scary ruling.

There is one more adjacency worth flagging, because it sits right next to this trade on the case map: **merger review**. The same antitrust apparatus that polices existing monopolies also gates *future* M&A, and a hostile enforcement posture chills the deal-making that platforms use to expand — every acquisition now risks a Meta-style "kill-zone" challenge years later. That changes how platforms allocate capital (more build, less buy) and creates a separate, tradeable risk in the *targets* they might otherwise have acquired. Sizing the deal-approval probability from a merger-arbitrage spread is its own discipline; the mechanics are in the [hedge-funds](/blog/trading/hedge-funds) series, and the broader question of how merger review and national-security screening gate cross-border deals connects back to the antitrust foundations.

The throughline of the whole series applies here in its purest form: a *rule* (or a ruling) changes the *rules of the game* for the most valuable companies on earth; the market discounts the *expected* change into prices long before any remedy bites; and the practitioner who has mapped each case to its revenue line, priced the remedy scenarios, and watched the multi-year timeline can read the overhang and size the trade while the consensus is still arguing about the fine.

## Further reading & cross-links

Within this series:

- [Antitrust 101: Sherman, Clayton, and merger review](/blog/trading/law-and-geopolitics/antitrust-101-sherman-clayton-and-merger-review) — the statutory foundations, per-se vs rule-of-reason, and how the consumer-welfare standard works before the platform cases stress it.
- [Regulatory risk as an asset-pricing factor](/blog/trading/law-and-geopolitics/regulatory-risk-as-an-asset-pricing-factor) — how a legal threat becomes a discount-rate haircut and a multiple compression, the engine behind everything in this post.
- [Data and privacy law: GDPR and the tech business model](/blog/trading/law-and-geopolitics/data-and-privacy-law-gdpr-and-the-tech-business-model) — the parallel squeeze on the ad-tech model from privacy rules, which compounds the antitrust overhang.
- [How law moves markets: the transmission chain](/blog/trading/law-and-geopolitics/how-law-moves-markets-the-transmission-chain) — the master spine the whole series hangs on.
- [How a rule becomes a price: expectations, drift, and repricing](/blog/trading/law-and-geopolitics/how-a-rule-becomes-a-price-expectations-drift-and-repricing) — the event-study toolkit for the ruling-day moves.
- [The regulatory calendar: trading the rulemaking clock](/blog/trading/law-and-geopolitics/the-regulatory-calendar-trading-the-rulemaking-clock) — building the catalyst calendar of trial dates, ruling windows, and DMA deadlines.

Cross-asset and valuation mechanics:

- [Equity research](/blog/trading/equity-research) — the DCF, multiples, sum-of-the-parts, and margin analysis used in every worked example above.
- [Cross-asset](/blog/trading/cross-asset) — index concentration, factor exposure, and how a mega-cap repricing transmits to the broad market.
