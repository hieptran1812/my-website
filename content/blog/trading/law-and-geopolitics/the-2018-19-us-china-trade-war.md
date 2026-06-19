---
title: "The 2018-19 US-China trade war: how tariffs repriced markets"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "A neutral, practitioner walkthrough of how the 2018-19 US-China tariff escalation moved equities, commodities, and the yuan in real time, and how to read the headline cycle."
tags: ["trade-war", "tariffs", "section-301", "us-china", "geopolitics", "regulation", "yuan", "supply-chain", "event-trading", "macro"]
category: "trading"
subcategory: "Law & Geopolitics"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — The 2018-19 US-China trade war is the cleanest modern case of trade *law* moving markets in real time: a Section 301 legal finding authorized tariff lists that pushed the average US tariff on Chinese goods from about 3% to about 19%, China retaliated, and every escalation headline and tariff list repriced equities, commodities, and the yuan within hours.
>
> - The mechanism is legal, not just economic: Section 301 of the Trade Act of 1974 let the executive impose tariffs by listing products, so the market had *dated, discrete events* (List 1, List 2, List 3, List 4) to price against.
> - A tariff is a tax on whoever cannot pass it on. US importers and consumers paid most of it; China bore the rest through lost orders and a weaker yuan. "China paid the tariffs" is the single biggest myth.
> - The trade war did not crash the market and it did not end with the deal: it *chopped* equities around the headline cycle, and most tariffs (near 19%) stayed in place long after the January 2020 Phase One truce.
> - The one number to remember: the average US tariff on Chinese imports went from **3.1% to 19.3%** and has stayed there for years. That durability is the real trade.

On the morning of **May 6, 2019**, a single message changed the price of nearly every risk asset on earth. Talks between Washington and Beijing had looked close to a deal for weeks; equity markets had drifted higher on the assumption that a truce was coming. Then a post on social media announced that the 10% tariff on \$200 billion of Chinese goods would be raised to 25%, effective that Friday, because — in the author's telling — China had tried to renegotiate. By the open, S&P 500 futures had gapped sharply lower. Over the next month the index fell roughly 6%, the Chinese yuan slid toward a level it had not touched in over a decade, soybean futures sagged, and semiconductor stocks led the way down. Nothing had *physically* changed: no factory had closed, no ship had been turned back. A legal threat had simply been repriced.

That is the whole point of this post. The 2018-19 US-China trade war is the defining modern laboratory for watching trade law move markets in real time. It is not a story about who was right. It is a story about *mechanism*: how a decades-old statute (Section 301 of the Trade Act of 1974) was used to impose tariffs through a sequence of published product lists; how each list, each threat, and each retaliation became a discrete, dated, tradeable event; how the yuan acted as a shock absorber that quietly handed back a chunk of the tariff; and how the supply chain began rerouting through Vietnam and Mexico in ways that are still reshaping markets today.

We will stay rigorously neutral. Whether the tariffs were good policy, whether they "worked," whether they helped or hurt particular workers — those are political questions, and they are not ours to settle. Our job is the practitioner's job: read the rule-change early, size the repricing, position in the winners and away from the losers, and know exactly what would prove the view wrong.

![Trade war transmission chain from a Section 301 finding to tariff lists to market repricing](/imgs/blogs/the-2018-19-us-china-trade-war-1.png)

The figure above is the whole post in one picture. A legal finding (lavender, the rule-maker) authorized tariff lists (blue, the instrument). China retaliated (red) and let the yuan slide (amber, the friction/offset). And at the end of every arrow sits a price: equities chopping, soybeans collapsing, semis selling off — until the Phase One deal (green) ended the *chop* but left the *tariff regime* almost entirely intact. Hold that chain in your head; every section below is one link in it.

## Foundations: how a tariff becomes a price

Before we touch a single chart, we need to build four things from zero: what a tariff actually is, where the legal power to impose one comes from, how an escalation becomes a market event, and why the currency matters. If you already trade this stuff, skim. If you are new, this is the load-bearing section — everything later assumes it.

### What a tariff actually is

A **tariff** is a tax on an imported good, collected at the border by the importing country's customs service. If the US puts a 25% tariff on a Chinese-made widget, then when that widget arrives at the port of Los Angeles, the US company importing it must pay the US government 25% of the widget's declared value before customs releases it. That is the entire mechanic. It is a tax, and like every tax, the crucial question is *who actually pays it* — which is almost never the same as who legally remits it.

Think of it like a toll booth on a bridge. The trucking company hands the cash to the toll operator, but whether that cost ends up borne by the trucker (thinner margin), the store the goods are headed to (higher wholesale price), or the shopper (higher shelf price) depends entirely on who has the power to push it onto the next party. That distribution — economists call it **tariff incidence** — is the difference between "China paid the tariffs" and "American importers and consumers paid the tariffs," and we will see the data says overwhelmingly the latter.

A tariff matters to markets for three reasons. First, it is a direct cost that lands on specific companies' income statements — an importer's gross margin literally shrinks by the tariff it cannot pass on. Second, it is a tax on trade volume, so it depresses the quantity of goods crossing the border, which hits exporters' revenue. Third, and most important for a trader, it is a *policy signal*: each new tariff says something about how far the conflict will escalate, and markets price the expected path, not just the current level.

One more mechanical detail makes the trade war legible, and it is worth a paragraph because it is *why the lists existed*. Tariffs are not applied to "China" as a blob; they are applied to specific products, and every product has a code in the Harmonized Tariff Schedule (HTS) — a standardized, internationally agreed numbering system that customs uses to classify everything from "8517.12.00 — telephones for cellular networks" to "1201.90.00 — soybeans." When the USTR built a tariff "list," what it literally published was a roster of HTS codes and the tariff rate to apply to each. That is why the lists could be precisely targeted (China's retaliation hit *soybean* HTS codes specifically, because that hurt politically) and why companies fought so hard over individual line items in the public comment process: getting your product's HTS code off the list, or onto an exclusion, was worth real money. For a trader, the HTS-code structure is what turned a vague "trade war" into a set of *measurable, attributable* costs you could map onto specific companies' cost of goods sold.

It also explains the *trade-weighted average* tariff number we keep citing. When you read "the average US tariff on Chinese goods is 19.3%," that is not a single rate — it is the dollar-weighted blend of zero-tariff goods, 7.5%-tariff goods, and 25%-tariff goods across the whole import basket. A product on List 3 might carry a full 25%; a product never listed carries the old ~3%; the *average* is what you get when you weight each by how much the US actually imports. That distinction matters because a company's pain depends on *its* product mix, not the headline average — a retailer sourcing entirely List 3 goods felt the full 25%, while a company importing mostly unlisted goods barely noticed.

### Where the legal power comes from: Section 301

Here is the part most market commentary skips, and it is the part that made the trade war *tradeable*. A US president cannot simply wake up and tax imports. The power to set tariffs belongs constitutionally to Congress. But over the 20th century, Congress delegated chunks of that power to the executive branch through specific statutes, each with its own trigger and its own procedure. The trade war ran almost entirely on one of them: **Section 301 of the Trade Act of 1974**.

Section 301 gives the United States Trade Representative (USTR) — an executive-branch agency — the authority to investigate whether a foreign country's trade practices are "unreasonable or discriminatory" and burden US commerce, and, if so, to retaliate, including by imposing tariffs. In August 2017 the USTR opened a Section 301 investigation into China's policies around intellectual property, technology transfer, and innovation. In March 2018 it published its findings: it concluded that China's practices were unfair under the statute. That legal finding is the headwaters of the entire river. It is the lavender box in our first figure. Everything downstream — every tariff list, every dollar of repricing — flows from that one administrative document.

Two other statutes played supporting roles, and it is worth naming them so you can recognize the tool by its fingerprint:

- **Section 232 of the Trade Expansion Act of 1962** lets the president impose tariffs on national-security grounds. The 2018 steel (25%) and aluminum (10%) tariffs came through Section 232. These hit *all* countries, not just China, and they are why "domestic steel" shows up as a winner later.
- **The International Emergency Economic Powers Act (IEEPA)** lets the president regulate commerce after declaring a national emergency. It was not the main engine in 2018-19, but it is the legal basis you will see invoked for the fastest, broadest tariff threats, because it has the loosest procedural guardrails.

The reason the statute matters to a trader is *speed and predictability*. Section 301 tariffs follow a process — a finding, a proposed product list, a public comment period, a final list, an effective date. That process creates a *calendar*. A comment period closing is a date. An effective date is a date. A threatened list becoming an imposed list is a date. Markets can position around dates. Compare that to a pure IEEPA emergency tariff, which can land overnight with little warning — that is far harder to trade and tends to produce bigger gaps. (We cover this calendar-reading skill in depth in the companion post on [the legal architecture of global trade](/blog/trading/law-and-geopolitics/the-legal-architecture-of-global-trade); here we just need to know the lists existed and were scheduled.)

### How an escalation becomes a market event

Markets do not wait for a tariff to take effect to reprice. They price the *expected* path of policy the moment new information arrives — a mechanism we walk through in detail in [how a rule becomes a price](/blog/trading/law-and-geopolitics/how-a-rule-becomes-a-price-expectations-drift-and-repricing). In the trade war, "new information" arrived in a recognizable taxonomy:

- A **threatened** list (announced but not yet imposed) moves markets on the *probability* it becomes real and on its *size*.
- An **imposed** list (now in effect) moves markets on the realized cost, but often *less* than the threat did, because the threat already priced in much of it — the classic "buy the rumor, sell the news" pattern works in reverse here ("sell the threat, buy the imposition").
- An **exclusion** (a product carved out, or a rate cut) is a de-escalation and rallies the affected names.
- A **retaliation** by China is a fresh escalation and reignites risk-off.

So the trade war was not one event. It was a *cycle*: escalate, risk-off; signal talks, rally; break down, risk-off again; truce, rally. The equity index did not trend so much as it *chopped* — a sawtooth driven by the headline flow. Internalize that word, *chop*, because it dismantles the "the trade war crashed the market" myth and it is the single most important thing to know if you ever try to trade this kind of headline cycle.

There is a subtlety in *how much* each headline moved markets, and it follows directly from the discounting logic. The market does not move on the *event*; it moves on the *surprise* — the gap between what happened and what was already priced. By mid-2019, the market had developed a "trade-war prior": a probability distribution over how the next round of talks would go, built from months of pattern-matching the headline flow. A headline that confirmed the prior (another threatened list when escalation was expected) moved markets little. A headline that *violated* the prior (the May 2019 hike, after weeks of optimism that a deal was imminent) moved them violently. This is why the same nominal escalation could produce a 6% drawdown one month and a shrug the next — the size of the move was set by the distance between the news and the consensus, not by the dollar value of the tariff.

For a practitioner this has a sharp implication: the tradeable edge was never in *predicting* the headlines (you cannot front-run a social-media post) but in *reading the positioning*. When the market had drifted complacent — low volatility, options cheap, equities priced for a deal — the asymmetry favored buying cheap protection ahead of a known catalyst, because any negative surprise would be amplified by the unwinding of crowded "deal is coming" trades. When the market was already panicked — high volatility, options expensive, everyone hedged — the asymmetry favored fading the fear, because any constructive headline would trigger a violent relief rally as hedges were unwound. The trade war, in other words, was a *positioning* game dressed up as a *geopolitics* game. The geopolitics set the catalysts; the positioning set the payoffs.

### Why the currency matters: the yuan as a shock absorber

The last foundation is the one that surprises beginners most. The price an American importer pays for a Chinese good is set in *dollars*, but the good is produced and priced in *yuan* (renminbi, CNY). The bridge between them is the USD/CNY exchange rate. If the yuan weakens — meaning it takes *more* yuan to buy one dollar, equivalently *fewer* dollars to buy a given pile of yuan-priced goods — then the dollar cost of that Chinese good falls, even before any tariff.

This is why the yuan became the trade war's most-watched single variable. China's central bank, the People's Bank of China (PBoC), manages the yuan within a band rather than letting it float freely (the legal mechanics of a managed currency are their own subject; if you want the full toolkit, see the macro series on monetary plumbing). During the trade war, the yuan steadily weakened, and in **August 2019** it broke through the psychologically and politically charged level of **7 yuan per dollar** for the first time since the 2008 crisis. The US Treasury promptly named China a "currency manipulator." Markets read the break of 7 as both an escalation (an implicit weapon) and an offset (a cheaper yuan refunds part of the tariff to US buyers). We will quantify exactly how much it offset.

With those four foundations — tariff incidence, the Section 301 legal engine, the escalate/de-escalate cycle, and the yuan offset — we can now read the actual history as a market practitioner.

## The escalation timeline and the market reaction at each step

The cleanest way to see trade law moving markets is to walk the timeline and mark the price at each step. The tariff path is not a smooth ramp; it is a staircase, and each step is a published list.

![Escalation timeline from Section 301 and List 1 through the yuan breaking 7 to the Phase One deal](/imgs/blogs/the-2018-19-us-china-trade-war-4.png)

Walk the staircase from the left:

**March-July 2018 — the finding and List 1.** The USTR published its Section 301 findings in March 2018. List 1 followed: a 25% tariff on roughly \$34 billion of Chinese imports, effective July 6, 2018, focused on industrial and tech-related goods. China retaliated with tariffs on a matching \$34 billion of US goods, deliberately targeting politically sensitive American exports — above all, **soybeans**. Markets wobbled but did not break; the dollar amounts were small relative to the \$500+ billion the US imported from China annually, and many investors still expected a quick deal.

**August-September 2018 — Lists 2 and 3.** List 2 added 25% on another \$16 billion. List 3 was the escalation that mattered: 10% on \$200 billion of imports, effective September 2018, with a threat to raise it to 25%. This was the moment the war went *broad* — it reached far enough into the import basket to touch consumer-adjacent goods and to start mattering to corporate guidance. Risk assets sold off on the announcement, and China retaliated again.

**May 2019 — the List 3 rate hike.** This is the May 6 episode from our opening. The 10% on \$200 billion was raised to 25% after talks broke down. The S&P 500 fell roughly 6% over the month. This step is the clearest single proof of the mechanism: no economic data had changed; a *legal rate* had changed, and that alone moved trillions in market value.

**August 2019 — the yuan breaks 7.** As the US threatened List 4 (tariffs on essentially all remaining imports, including consumer staples like phones, toys, and apparel), the yuan slid past 7 per dollar. The Dow fell about 767 points on the day of the break. This was the war's emotional peak: a currency level breaking and a "manipulator" label, layered on top of the tariff threat.

**September 2019 — List 4A imposed.** A 15% tariff on about \$112 billion (List 4A) took effect, finally reaching shelves — this is when the tariff began touching the goods ordinary consumers buy directly. A second tranche (List 4B, on the most sensitive consumer goods) was scheduled for December.

**January 2020 — Phase One.** Negotiators signed a "Phase One" agreement. List 4B was canceled. List 4A's rate was cut from 15% to 7.5%. China pledged to buy an additional \$200 billion of US goods and services over two years. Crucially — and this is the durability point we will hammer in the misconceptions section — the Lists 1, 2, and 3 tariffs (the bulk, at 25% on roughly \$250 billion) *stayed*. The chop ended. The tariff regime did not.

Now put numbers on that staircase. The data series below is the trade-weighted average US tariff on Chinese goods, and its mirror image, the average Chinese tariff on US goods, from the PIIE/Chad Bown tariff tracker.

![US tariff on China and China tariff on US dual line chart 2018 to 2024](/imgs/blogs/the-2018-19-us-china-trade-war-2.png)

Two things jump out. First, the symmetry: this was genuinely *tit-for-tat*. The US average climbed from 3.1% to roughly 21% at the peak and settled near 19.3%; China's climbed from 8.0% to roughly 21% and settled near 21.1%. Each side matched the other's escalation step for step. Second, the flatline on the right: after Phase One in early 2020, *neither side meaningfully cut*. The tariff wall that was built in 18 months has stood for years. The trade was never just the escalation — it was the permanence.

It is equally important to see what the trade war was *not*. It was not a general turn toward protectionism across all trading partners. It was a wall between two specific countries. The chart below contrasts the tariff each side imposed *on the other* with the tariff each imposed *on the rest of the world*.

![Bar chart contrasting bilateral US China tariffs with tariffs on the rest of the world](/imgs/blogs/the-2018-19-us-china-trade-war-3.png)

The gap is the whole story of the rerouting trade. The US tariff on Chinese goods sits at 19.3%, but the US tariff on goods from everywhere else is still about 3%. China's tariff on US goods is 21.1%, but its tariff on the rest of the world is about 6.5%. That ~16-point wedge between "from China" and "from anywhere else" is precisely the incentive that pushed factories to Vietnam and Mexico — a producer could shave roughly 16 points of tariff off its US-bound goods simply by changing the country stamped on the box. A trade war that taxed *all* imports equally would not have created beneficiaries; because this one taxed only one origin, it created an entire investable theme out of the arbitrage between origins. Keep that wedge in mind — it is the engine of the supply-chain section later.

#### Worked example: the tariff cost on a \$500 imported product

Take a concrete US retailer importing a consumer electronics accessory from China. The landed cost (the price it pays the Chinese factory, before tariff) is \$500 per unit. List 3 puts a 25% tariff on it.

The tariff owed at the border is `25% × $500 = $125` per unit. The retailer's all-in cost rises from \$500 to **\$625**. Now suppose this retailer was selling the unit at \$700 with a gross margin of `($700 − $500) / $700 = 28.6%`. If the retailer *cannot* raise its price (competitive market, price-sensitive shoppers), its new gross margin is `($700 − $625) / $700 = 10.7%` — its gross profit per unit has collapsed from \$200 to \$75, a **62.5% drop in unit gross profit**. To *restore* the old \$200 gross profit, it would have to raise the shelf price to \$825 — an 18% price increase the shopper would feel. The one-sentence intuition: a 25% tariff on the *input* is a far bigger than 25% hit to the *profit*, because margins are thin and the tariff eats the margin, not the revenue.

## Sector winners and losers: where the tariff landed

A tariff is not a uniform weight on the whole market. It is a redistribution. It taxes whoever sits on the wrong side of the border and shields whoever competes with the now-taxed imports. The practitioner's edge is mapping that redistribution before the market fully prices it.

![Matrix of trade war sector winners and losers by mechanism and market mark](/imgs/blogs/the-2018-19-us-china-trade-war-5.png)

Read the matrix top to bottom, mechanism first:

**Importers and retailers — losers.** Big-box retailers, furniture sellers, apparel companies, and machinery importers sourced heavily from China and operated on thin margins. They paid the tariff at the port and, in competitive categories, could not fully pass it on. The worked example above is their story. Their guidance got cut; their multiples de-rated.

**US soybean farmers — losers, then subsidized.** China was the single largest export market for US soybeans. China's retaliation deliberately targeted them, because soybeans are grown in politically pivotal states. US soybean exports to China fell roughly 75% in 2018, and the soybean price dropped about 20% in the weeks after the retaliation. The US government responded with a **farm-aid program** (the Market Facilitation Program) of tens of billions of dollars to offset farmers' losses — a fiscal transfer that itself became a market and political fact.

**Semiconductors — losers (and the most volatile).** Chipmakers and chip-equipment firms had large revenue exposure to China both as a *market* and as a node in the *supply chain*. The semiconductor index (SOX) became a real-time barometer of trade-war sentiment, whipsawing sharply on each headline. Semis were the "high beta" of the trade war: when risk-off hit, they fell hardest; when a truce was signaled, they ripped.

**Domestic steel — winner (via a different statute).** Here the legal detail matters. The Section 232 steel tariffs (25% on imported steel, applied broadly) *shielded* US steelmakers from foreign competition. Domestic mills got an early pop and announced some capacity restarts. But — and this is the honest, neutral read — the benefit faded as input costs rose and demand softened; the steel "winner" trade was real but not durable.

**Vietnam and Mexico — winners (the rerouting trade).** This is the second-order effect that outlasted the headlines. Because the tariff was on goods *from China specifically*, the cheapest way to dodge it was to produce somewhere else. Orders, and then factories, began shifting to Vietnam and Mexico. We give this its own section because it is the most investable durable consequence — and the bridge to the Vietnam story we tell in full elsewhere.

A note on *why* the matrix is asymmetric in a way that matters for sizing: the losers are concentrated and visible, but the winners split into two very different kinds of bet. The "shielded domestic producer" winner (steel) is a *policy-dependent* trade — it lives and dies by the tariff staying in place and by the protected industry actually converting protection into profit, which steel largely failed to do as input costs and soft demand caught up. The "rerouting beneficiary" winner (Vietnam, Mexico) is a *structural* trade — it does not need the tariff to rise further, only to *persist*, because relocating a supply chain is a multi-year capital decision that compounds long after the headlines fade. A practitioner who lumped both winners together would have been disappointed by steel and underweight the durable theme. The lesson generalizes: when a rule creates winners, separate the ones who win *because the rule exists* from the ones who win *because the rule changed behavior* — the second group is where the durable money sits.

### The macro feedback: how the trade war pulled the Fed

The deepest second-order effect ran through monetary policy, and it is the link most retail commentary missed. Tariffs and trade uncertainty are a drag on business investment: when a company does not know whether its supply chain will be taxed next quarter, it delays building the new plant, postpones the big order, sits on cash. Through 2019, that uncertainty showed up as softening manufacturing surveys (the ISM manufacturing index dipped below 50, into contraction) and slowing capital expenditure. The Federal Reserve, watching growth wobble with inflation still tame, responded: it cut interest rates three times in the second half of 2019 — the so-called "mid-cycle adjustment" — explicitly citing trade-policy uncertainty and weak global growth among its reasons.

This closes a feedback loop that is easy to miss and crucial to trade. The trade war was *risk-off* for equities through the tariff and earnings channel — but it was simultaneously *risk-on* for equities through the rate channel, because the drag it created pushed the Fed to ease, and easier policy lifts asset prices. That tension is a big part of why the index *chopped* rather than crashed: every escalation that hurt earnings expectations also raised the odds of a Fed cut that supported valuations. The two forces fought to a draw at the index level, even as they tore individual sectors apart. If you want the full mechanism of how a rate cut transmits into asset prices, the macro series covers it; here the point is just that *the same trade-policy shock hit markets through two opposing channels at once*, and the net was chop.

#### Worked example: the trade-war drag translated into a Fed cut and an equity offset

Make the feedback concrete with a stylized valuation. Take a stock earning \$5.00 per share, trading at a price-to-earnings multiple of 18, so its price is `$5.00 × 18 = $90.00`.

Now hit it with the trade war through both channels. Channel one (earnings): tariff costs and softer demand shave 4% off expected earnings, to `$5.00 × 0.96 = $4.80`. Channel two (rates): the trade-war drag pushes the Fed to cut, the discount rate falls, and the multiple the market is willing to pay expands from 18 to 18.8 (about +4.4%). The new price is `$4.80 × 18.8 = $90.24`.

The stock is almost exactly flat — `$90.24` versus `$90.00` — despite a real, measurable hit to earnings. The earnings damage (−4%) was offset by the multiple expansion (+4.4%) that the *same shock* produced via the Fed. The one-sentence intuition: the trade war did not crash the index because the policy drag it caused triggered a monetary offset, and a trader who modeled only the earnings channel would have been short into a market that refused to fall.

#### Worked example: a soybean farmer's revenue loss and the farm-aid offset

Take a Midwest farm that grows 100,000 bushels of soybeans a year and, before the war, sold the bulk to buyers serving the Chinese market at about \$10.00 per bushel — roughly \$1,000,000 in revenue.

China's retaliatory tariff makes US soybeans uncompetitive in China; the US soybean price falls about 20%, to roughly \$8.00 per bushel. If the farm still sells all 100,000 bushels but at the lower price, revenue drops to `100,000 × $8.00 = $800,000` — a **\$200,000 loss** purely from the price move. Worse, some volume goes unsold or into storage because the largest buyer has effectively exited.

Now the offset: suppose the farm receives a farm-aid payment of \$1.65 per bushel (a representative Market Facilitation Program rate), or `100,000 × $1.65 = $165,000`. That recovers most — but not all — of the \$200,000 price loss, leaving the farm roughly \$35,000 worse off on price alone, before counting unsold volume and storage cost. The one-sentence intuition: retaliation transferred the pain from a foreign tariff into a domestic fiscal subsidy, so the farmer was partly made whole by the taxpayer rather than by the market — a fact that itself shaped the politics and the durability of the tariffs.

## The yuan offset: how much did a weaker currency neutralize?

This is the most underappreciated part of the whole episode, and it is where a practitioner who understands the mechanics has a real edge over headline readers. The tariff was a tax on Chinese goods; the falling yuan was a discount on Chinese goods. They partly cancel.

![Pipeline showing a weaker yuan offsetting the dollar cost of a tariff](/imgs/blogs/the-2018-19-us-china-trade-war-6.png)

The logic, step by step from the figure: a tariff is imposed (25%, at the US port). The yuan weakens (the PBoC lets it slide as trade pressure builds). Because the good is priced in yuan, a weaker yuan means the *dollar* price of the same good falls. So the *net* cost felt by the US buyer is the tariff *minus* the currency discount. A 25% tariff with a 10% currency offset feels more like a 15% tariff.

Let us be precise about the arithmetic, because the direction of the exchange rate trips people up.

#### Worked example: how a yuan move from 6.3 to 7.0 offsets a 25% tariff

Take a Chinese good with a factory price of **¥630** (yuan). Walk the dollar cost through two exchange-rate regimes.

**Before the yuan weakened, at 6.3 yuan per dollar:** the pre-tariff dollar cost is `¥630 / 6.3 = $100.00`. Apply the 25% tariff: `$100.00 × 1.25 = $125.00`. The US importer pays \$125.00.

**After the yuan weakened to 7.0 yuan per dollar:** the same ¥630 good now costs `¥630 / 7.0 = $90.00` pre-tariff — the dollar price fell 10% purely from the currency. Apply the same 25% tariff: `$90.00 × 1.25 = $112.50`. The US importer now pays \$112.50.

So despite a 25% tariff, the US importer's cost rose from a *no-tariff, strong-yuan* baseline of \$100.00 to only \$112.50 — an effective increase of **12.5%**, not 25%. The currency depreciation gave back almost half of the tariff. Put differently: relative to the \$125.00 the importer "should" have paid, the weaker yuan refunded `$125.00 − $112.50 = $12.50` per \$100 of goods. The one-sentence intuition: a managed currency is a country's pressure-release valve on a tariff, which is exactly why the US watched the yuan so closely and why breaking 7 was treated as an act of escalation, not just a market move.

This offset is also why the tariffs were less inflationary in the US than a naive "25% on \$250 billion of imports" calculation would suggest, and why much of the academic work concluded that US importers and consumers — not Chinese exporters — bore the *tariff* cost, while the *currency* move quietly absorbed a meaningful slice of it on China's side. Both things are true at once, and holding both is what separates an analyst from a pundit.

## The supply-chain rerouting: Vietnam and Mexico

The tariff was a tax on *origin*, not on *demand*. American consumers still wanted the same phones, furniture, and footwear. The cheapest way to satisfy that demand without paying the China tariff was to make the goods somewhere the tariff did not apply. So the supply chain began to reroute — slowly at first, then structurally.

![Supply chain rerouting from China to Vietnam and Mexico to enter the US market](/imgs/blogs/the-2018-19-us-china-trade-war-7.png)

The before/after is the cleanest way to see it. *Before 2018*, goods flowed straight from China — the factory of the world — into the US market at a low (~3%) tariff; China's scale set the price and tariffs barely entered the calculus. *After 2019*, China itself faced a 19-25% US tariff, so final assembly migrated. **Vietnam** became the marquee beneficiary in light manufacturing, electronics assembly, and furniture; **Mexico** picked up nearshoring, helped by preferential access to the US market under the USMCA agreement. The same demand was met from a new origin.

The data backs the story. Vietnam's foreign direct investment — the money companies commit to building and expanding factories there — climbed in the years around and after the trade war, with realized (disbursed) FDI rising steadily even as global investment was choppy.

![Vietnam FDI registered versus disbursed rising after the trade war](/imgs/blogs/the-2018-19-us-china-trade-war-8.png)

Notice the gap between *registered* (committed) and *disbursed* (actually deployed) FDI: commitments are a leading indicator, disbursements are the lagging confirmation. A practitioner watching the rerouting trade tracks both — registered tells you intent, disbursed tells you the factories are actually going up. The steady climb in disbursed FDI from about \$20 billion to \$25 billion is the rerouting trade made physical.

The corporate name for this shift is "China+1" — a strategy where a multinational keeps its Chinese capacity but adds a second base elsewhere to diversify tariff and geopolitical risk. The key word is *plus*: this was rarely a wholesale exit from China (China's scale, infrastructure, and skilled workforce remain hard to replicate) but an *incremental* relocation of the most tariff-exposed, most labor-intensive, or most politically sensitive production. That nuance matters for sizing the trade. A naive thesis ("China loses everything, Vietnam wins everything") overshoots; the realistic thesis is a gradual, multi-year *redistribution at the margin* that compounds. The investable consequence widens beyond the destination country: the *logistics* firms that move the rerouted goods, the *industrial-real-estate* developers that build the new factories and warehouses, the *automation and capital-equipment* makers that tool them, and the *commodity and component* suppliers feeding the new plants all capture a slice. The rerouting trade, properly understood, is a basket — destination equities, the multinationals relocating, and the picks-and-shovels of the rebuild — not a single bet on one flag.

A crucial honesty caveat, because this is where the rerouting trade gets dangerous: a lot of early "Vietnam" exports were really *transshipment* — Chinese goods routed through Vietnam with minimal local value added, to relabel the origin and dodge the tariff. US customs authorities know this and have increasingly scrutinized rules-of-origin, slapped anti-circumvention duties on specific products, and watched Vietnam's ballooning trade surplus with alarm (Vietnam was briefly labeled a currency manipulator in 2020). So the rerouting beneficiary trade carries its own tail risk: the same tariff logic that creates the beneficiary can, with one rule change on origin, turn around and tax it. We unpack that two-sided bet in full in [Vietnam in the US-China squeeze](/blog/trading/law-and-geopolitics/vietnam-in-the-us-china-squeeze).

#### Worked example: a reshoring beneficiary's margin gain

Take a Vietnamese contract manufacturer that assembles a product for a US brand. Before the trade war, the US brand sourced the same product from a Chinese factory at a landed cost of \$50 per unit. After List 3, that Chinese-sourced unit carries a 25% tariff: `$50 × 1.25 = $62.50` landed in the US.

The Vietnamese factory can produce the same unit at a slightly higher base cost — say \$53, because Vietnam's supply ecosystem is less mature — but it faces no China tariff (Vietnam's normal US tariff on the category is low, assume ~3%): `$53 × 1.03 = $54.59` landed. The US brand now saves `$62.50 − $54.59 = $7.91` per unit, about **12.7%**, by switching origin — *even though Vietnam's factory-gate cost is higher*.

For the Vietnamese manufacturer, the new orders are pure incremental volume. If it wins 1,000,000 units of annual demand at a \$4 contribution margin per unit, that is `1,000,000 × $4 = $4,000,000` of new annual gross profit it did not have before the tariff existed. The one-sentence intuition: the tariff did not create new demand, it *relocated* the profit — handing it to the producer in the country the tariff does not touch, which is the entire reshoring/friend-shoring investment thesis in one number.

## Common misconceptions

The trade war generated more confident, wrong statements per capita than almost any market event of the era. Three myths matter most, and each one falls apart against a number.

**Myth 1: "China paid the tariffs."** No. The tariff is legally remitted by the *US importer* at the US border. The economic question is incidence — who bears the cost — and the bulk of the academic evidence (the New York Fed, and economists like Amiti, Redding, and Weinstein) found that US importers and consumers bore essentially *all* of the tariff cost in the form of higher prices: the pre-tariff prices of Chinese goods barely fell, meaning Chinese exporters did not "eat" the tariff by cutting their prices. Our \$500-product worked example is the micro version: the \$125 tariff hit the US retailer's margin or the US shopper's wallet. China bore costs too — but through *lost export orders* and a *weaker currency*, not by paying the US tariff. The slogan conflates "who writes the check to customs" with "who is poorer at the end," and they are not the same party.

The way researchers established this is itself instructive for a trader, because it is the same logic you would use to test any "who pays the tax" claim. If Chinese exporters had absorbed the tariff, you would see the *pre-tariff border price* of Chinese goods fall by roughly the tariff amount — they would cut their prices to keep their goods competitive after the tax. The studies found those border prices were essentially *flat*: the full tariff showed up on top of an unchanged base price, which is the signature of *near-complete pass-through to the importer*. The neutral, evidence-based read is therefore not a slogan in either direction: the *tariff* was paid by Americans, the *trade loss* (fewer orders, a weaker currency, slower export growth) was borne by China, and both are real. A practitioner who understood this did not waste time arguing the politics; they used the pass-through finding to correctly mark down the *importers'* margins rather than the *exporters'*.

**Myth 2: "The Phase One deal ended the tariffs."** No. Phase One canceled the *threatened* List 4B and halved List 4A (from 15% to 7.5%), but it left Lists 1, 2, and 3 — the 25% tariffs on roughly \$250 billion of goods — fully in place. The average US tariff on Chinese imports settled at **19.3%** and, as our dual-line chart shows, stayed there through 2024. The deal ended the *escalation*, not the *tariffs*. A trader who sold protection or bought beneficiaries expecting a tariff *rollback* would have been wrong for years.

**Myth 3: "The trade war crashed the market."** No. It *chopped* it. The S&P 500 had sharp, headline-driven selloffs (about −6% in May 2019, a brutal Q4 2018 that had multiple causes including Fed policy) but also sharp truce-driven rallies, and it ended 2019 up strongly — one of the best equity years of the decade. The pattern was a sawtooth, not a cliff. Confusing the two leads to the worst possible trade: buying expensive downside protection and holding it through the rallies, bleeding premium while the index grinds higher between headlines. The trade war was a *volatility* event far more than a *direction* event.

A fourth, subtler myth deserves a mention: "the tariffs were quickly reversed by the next administration." They were not. The durability of the tariffs across a change in administration is one of the most important facts for an investor, because it tells you this is a *bipartisan, structural* policy, not a personal one — and structural policy is the kind you position around for years, not weeks. We return to this in the playbook.

The neutral way to read that durability is as information about the *underlying preference*, not the *personality*. When a policy survives a change in the party controlling it, the market should update toward "this reflects a durable national-interest consensus" and away from "this is a reversible negotiating tactic." That single update is worth more than any individual headline, because it resets the *horizon* of every downstream trade: the reshoring beneficiary thesis, the de-risking of supply chains, the strategic competition framing of technology. A trader who treated the tariffs as a personality-driven aberration kept waiting for a reversal that never came and missed years of the structural theme; a trader who read the bipartisan durability correctly held the right trade through multiple political cycles. The lesson is general and it is the quiet thesis of this whole series: when a rule outlives the people who wrote it, treat it as part of the landscape, not as weather.

## How it shows up in real markets

Strip away the narrative and four concrete market signatures recur whenever the trade-war headline machine turns on. These are the patterns you would actually see on a screen.

**The escalate/de-escalate chop in equities.** Index-level price action during 2018-19 was a function of the headline calendar more than of earnings or the economy. A threatened list, a broken-off negotiation round, or a fresh retaliation produced a one-to-three-day risk-off air pocket — gap down, vol spike, defensive sectors outperform. A signaled truce, a "constructive call," or an exclusion produced the mirror-image relief rally. The professional read was that *realized direction over months was near zero while realized volatility was high*: the definition of chop.

**The soybean collapse.** Agricultural commodities, soybeans above all, were the cleanest single-name expression of China's retaliation. Because China's tariff was targeted and China was the dominant buyer, the soybean price reacted almost mechanically to retaliation headlines and to "China is buying again" headlines. Soybeans were, for two years, a trade-war sentiment instrument that happened to also be a crop.

**The yuan-7 break.** USD/CNY was the macro tell. As long as the yuan was stable, the conflict was "contained." When the PBoC let it drift weaker, it signaled either a deliberate offset or a loss of control over capital outflows, and risk assets globally took their cue from it. The August 2019 break of 7 was the loudest single bar of the whole episode: it moved the Dow, US Treasuries (lower yields, a flight to safety), and gold (higher, a safe haven) simultaneously. When one currency level moves three asset classes at once, that level *is* the trade.

**The semis selloff.** The Philadelphia Semiconductor Index (SOX) was the equity-market amplifier. High China revenue exposure plus supply-chain entanglement plus a growth-stock multiple made semis the highest-beta expression of trade-war risk. If you wanted to express a trade-war *view* with leverage, you did it in semis; if you wanted to *read* trade-war sentiment, you watched semis relative to the broad market. The reason semis amplified so cleanly is worth unpacking: a chip company is exposed to China three ways at once — as an *end market* (a large share of revenue is sold into Chinese electronics manufacturing), as a *supply-chain node* (assembly, test, and packaging often happen in or near China), and as a *long-duration growth asset* (its valuation rests on far-future earnings, which a rising discount rate or a demand shock hits hardest). Stack those three exposures and you get a sector that moves several times the index on the same headline. That is why a desk would use the SOX, or a liquid semiconductor ETF, as both a sentiment gauge and a high-conviction expression vehicle — and why, when the export-control phase of the conflict arrived later, semiconductors moved from "high-beta proxy" to "the actual battlefield."

There is a fifth signature that is easy to overlook because it is an *absence*: the goods that were never listed barely moved. Companies whose import basket sat outside the tariff lists — services firms, domestic-demand businesses, importers of unlisted categories — traded through the whole episode with little trade-war beta. This is the practical payoff of the HTS-code structure from the foundations: the trade war was *attributable*, so you could build a long-short book that was long the no-exposure names and short the high-exposure names and harvest the dispersion without taking an index-level directional view. The dispersion *between* winners and losers was often a cleaner, higher-Sharpe trade than the direction *of* the index — a recurring truth about policy shocks that redistribute rather than uniformly tax.

These four signatures generalize. They are the same reaction map you see whenever a trade-policy shock hits — which is exactly why this case study is in a series about reading rules, and why we cross-reference the broader [event-trading mechanics](/blog/trading/event-trading) for the headline-reaction toolkit and the [macro-trading framework](/blog/trading/macro-trading) for how a policy shock transmits through liquidity and the dollar.

#### Worked example: sizing the May 2019 escalation as an event trade

Suppose on the evening of Sunday, May 5, 2019, you read the tariff-hike threat and want to size a trade. The S&P 500 is at roughly 2,930. History says a serious escalation headline tends to produce a 2-6% drawdown over the following weeks if it sticks, and a near-full recovery if it is a bluff.

Assign rough probabilities: 60% it sticks (talks really have broken down), 40% it is a negotiating bluff that fades within days. If it sticks, estimate a −5% move (to ~2,785); if it fades, estimate roughly flat (call it +0.5%, to ~2,945). The expected move is `0.60 × (−5%) + 0.40 × (+0.5%) = −3.0% + 0.2% = −2.8%`.

A trader holding \$1,000,000 of S&P exposure faces an expected hit of `2.8% × $1,000,000 = $28,000` if they do nothing. Hedging by reducing exposure 50% caps the expected hit near `1.4% × $1,000,000 = $14,000` while giving up half the upside if it fades. Whether to hedge depends on the *asymmetry*: here the downside (−5%) is roughly ten times the upside (+0.5%), so trimming or buying a defined-risk put spread is the rational move even though the *base case* is only a few percent. The one-sentence intuition: trade-war headlines are negatively skewed events — the loss if you are wrong dwarfs the gain if you are right — so you size for the skew, not for the midpoint.

## How to trade it: the playbook

Every post in this series ends on the same question: *so how do you actually read or trade this?* Here is the trade-war playbook, distilled. It is a framework, not advice — but it is the framework a desk would actually use.

Start from the spine: law/geopolitics changes the rules of the game, markets discount the change before it fully bites, and the practitioner reads the change early, sizes the repricing, and knows what invalidates the view. The trade war is the textbook instance because every link in that chain was *observable*. The rule-change was a published USTR document with a known process. The discounting was visible in how the index priced threats more than impositions. The repricing was attributable, sector by sector, through the HTS-code structure. And the invalidation conditions were knowable in advance. When a market event hands you all four — a legible rule, a measurable discount, attributable winners and losers, and a clear kill switch — you do not need an information edge to trade it well; you need a *framework* edge, and that is what the five rules below encode.

**1. Trade the escalation/truce headline cycle, not the direction.** The single biggest mistake was treating the trade war as a directional bet ("the market will crash"). It was a *volatility regime*. The professional posture was to be roughly market-neutral on direction and to monetize the chop: sell expensive protection into truce rallies, buy cheap protection when complacency set in before a known catalyst (a deadline, a negotiation round, an effective date). The catalyst calendar — when does the next list take effect, when does the next round of talks happen — was the edge, the same way an earnings calendar is.

The right *instrument* for that cycle is usually options, not the underlying, because options let you isolate the volatility from the direction. Ahead of a known binary catalyst (a tariff deadline, a high-stakes negotiation round), a defined-risk structure — a put spread, or a strangle if you genuinely do not know the direction — caps your cost and your loss while giving you convex exposure to a gap. Selling that same volatility *after* the event, into the relief, harvests the inflated premium. The trap to avoid is owning naked, far-dated protection through the whole regime: the chop means you bleed time-decay (theta) on every rally between headlines, and the premium you pay over months can dwarf the protection you ever cash in. Match the *tenor* of the hedge to the *catalyst*, not to the whole conflict.

**2. Position in beneficiaries, not just away from cost-bearers.** The losing trade was visible (importers, exposed exporters, high-China-revenue semis), but the *durable* money was in the beneficiaries of the structural shift: reshoring and friend-shoring plays (Vietnam, Mexico, India equities and the multinationals relocating capacity), domestic producers shielded by tariffs (with the caveat that the Section 232 steel benefit faded), and the automation/logistics names that profit when supply chains get rebuilt. The losers reprice in weeks; the beneficiaries re-rate over years. A subtle point on the beneficiary trade: the cleanest expression was often *not* the obvious local-market ETF (Vietnam's small, foreign-ownership-capped market is hard to access at scale) but the *multinationals* visibly relocating capacity and the *capital-goods* firms selling the factory equipment for the rebuild. Follow the capex, not just the flag on the map.

**3. Watch the yuan as the master tell.** USD/CNY was the highest-information single variable. A stable yuan said "contained." A weakening yuan said "China is offsetting and/or escalating." A break of a round, watched level (7.00) said "regime change in the conflict." If you could only watch one screen during the trade war, it was USD/CNY, with gold and the 10-year Treasury yield as the confirming safe-haven instruments.

**4. Distinguish threatened from imposed.** Threats moved markets more than impositions, because the threat priced the *probability-weighted* future while the imposition was already discounted. The repeatable pattern was: fade the panic on the *threat* if the base rate said it was a negotiating tactic, and do *not* expect a second leg down when the tariff was actually imposed on schedule. "Sell the threat, buy the imposition" was a real edge for those who read the legal calendar.

**5. Respect the durability.** The deepest lesson is that the tariffs *stayed*. Once you recognized that this was bipartisan, structural policy — not a personality-driven, easily-reversed gesture — the right horizon for the beneficiary trade lengthened from months to years. The reshoring/friend-shoring theme was investable long after the headlines went quiet precisely *because* the tariff wall did not come down.

**What invalidates the view.** A playbook is only honest if it states its own kill switch. The trade-war framework is wrong, and you should exit or flip, if: (a) the tariffs are *actually rolled back* in a broad, durable way — that would invalidate the entire reshoring thesis and re-advantage the cost-bearers; (b) the yuan *strengthens* sharply and sustainably, removing the offset and the escalation tell; (c) the rerouting beneficiaries get hit by *origin/anti-circumvention* rules or their own tariffs (the Vietnam tail risk), which would turn the beneficiary trade into a new cost-bearer trade; or (d) the conflict broadens from tariffs into *export controls and investment screening* — a different, harder-to-offset weapon that hits specific technologies rather than broad goods, and which is its own case study. If none of those four things happen, the structural tilt — short the cost-bearers, long the rerouting beneficiaries, neutral-with-vol-harvesting on the index — remains intact.

The 2018-19 trade war is, in the end, the purest teaching case this series has. A single legal finding under a 1974 statute set off a sequence of dated, published events; each event repriced equities, commodities, and a currency in real time; the currency quietly refunded part of the tax; the supply chain rerouted in ways still playing out; and the rules, once changed, stayed changed. Master this one, and you have the template for every trade-policy shock that follows.

## Further reading and cross-links

Within this series:

- [The legal architecture of global trade](/blog/trading/law-and-geopolitics/the-legal-architecture-of-global-trade) — the WTO, MFN, FTAs, and the tariff statutes (Section 301/232/IEEPA): who can impose a tariff and how fast. The legal toolkit this case study runs on.
- [Vietnam in the US-China squeeze](/blog/trading/law-and-geopolitics/vietnam-in-the-us-china-squeeze) — the friend-shoring boom, transshipment scrutiny, and the two-sided beneficiary/tail-risk bet teased above.
- [How a rule becomes a price: expectations, drift, and repricing](/blog/trading/law-and-geopolitics/how-a-rule-becomes-a-price-expectations-drift-and-repricing) — the event-study mechanism behind "the market priced the threat before it bit."
- [How law moves markets: the transmission chain](/blog/trading/law-and-geopolitics/how-law-moves-markets-the-transmission-chain) — the spine of the whole series, of which this is one worked instance.

Cross-asset and mechanism context:

- [Macro trading](/blog/trading/macro-trading) — how a policy/liquidity shock transmits through the dollar, rates, and global flows.
- [Event trading](/blog/trading/event-trading) — the headline-reaction toolkit for trading discrete, dated catalysts like tariff lists and negotiation rounds.
