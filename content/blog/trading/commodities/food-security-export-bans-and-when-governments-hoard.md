---
title: "Food Security, Export Bans, and When Governments Hoard"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "How politics overrides the market when food gets scarce: the moment a price signal makes a government ban exports, why thin trade makes staples so fragile, and how that one decision multiplies a small crop shock into a global price spike."
tags: ["commodities", "food-security", "export-bans", "rice", "wheat", "agriculture", "geopolitics", "thin-markets", "vietnam", "stocks-to-use"]
category: "trading"
subcategory: "Commodities"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — When a staple food gets scarce, the price signal that is supposed to clear the market instead triggers a *political* reflex: the exporting government bans exports to protect its own people, and that single act multiplies the shortage into a global price spike.
>
> - **A price signal in food can flip from a market event into a political one.** A rising bread or rice price hits the poor first, so governments override the market — and an export ban is the most common override.
> - **Thin trade is the fragility.** Only about **8-9% of the rice the world grows ever crosses a border**; the rest is eaten where it is grown. So a small cut by one big exporter removes a *large* share of everything that is actually for sale.
> - **Bans cascade.** One ban shrinks world supply, frightens importers into panic-buying, invites copycat bans by neighbors, and the price spikes far more than the original crop shortfall ever justified. Rice **nearly tripled** in 2007-08 this way.
> - **Watch the exporters, the stocks-to-use ratio, and the political calendar.** The risk in staples is not a weather model — it is a policy decision, and you can see the pressure building before the announcement.
> - The one number to remember: in 2007-08, with only ~7% of world rice production traded, a wave of export restrictions sent the benchmark price from roughly **\$335 to nearly \$1,000 per tonne** — without any global rice *shortage* in the aggregate.

In the autumn of 2007 the world had plenty of rice. Stocks were lower than the bumper years, yes, but there was no famine, no continent-wide crop failure, no aggregate shortfall that could explain what happened next. And yet, over the following six months, the price of rice on the world market **nearly tripled** — from around \$335 a tonne to almost \$1,000 — and dozens of countries from Haiti to Bangladesh to Egypt erupted in food riots. The Prime Minister of Haiti was forced out of office. Governments fell. And the trigger was not a drought. The trigger was a *decision*.

It started with a relatively ordinary problem: wheat prices had spiked on a poor global harvest, and some governments, worried that their citizens would substitute into rice and bid it up, decided to get ahead of the problem. India restricted rice exports. Vietnam announced it would cut its export quota. The Philippines, the world's largest rice importer, panicked and put out enormous tenders to buy rice at almost any price. Each of those moves was rational for the country making it. Each protected someone's domestic consumers. And stacked on top of one another, in a market where only a sliver of the crop is ever for sale, they produced one of the most violent price spikes in the history of any commodity — in a year with no actual rice shortage. This is the single most important and least intuitive fact about food markets: **a staple can spike not because the world ran out, but because everyone tried to keep their own share.**

This post is about that mechanism. It is the moment where the clean logic of the rest of this series — where a price is a physical thing forced through a financial contract, and the curve and the carry decide who profits — runs straight into something the forward curve cannot price: a government that decides feeding its own people matters more than honoring a contract. We will build the cascade from zero, look at the real crises that prove it (2007-08 rice, 2010-11 wheat and the Arab Spring, 2022's war and the wave of bans, and India's market-reshaping 2023 rice ban), work the dollar math of why thin trade is so dangerous, and end on how a trader actually reads policy risk in staples.

![Export ban cascade pipeline from supply shock to global price spike](/imgs/blogs/food-security-export-bans-and-when-governments-hoard-1.png)

## Foundations: how a food market is different from every other commodity

Let us build this from zero, assuming you know nothing about agricultural markets. Most of this series treats commodities as standardized physical goods that trade on paper — a barrel of oil, a tonne of copper. (If that framing is new, the series opener on [what a commodity is](/blog/trading/commodities/what-is-a-commodity-the-physical-asset-that-trades-on-paper) walks through it.) A staple food — rice, wheat, the major grains — is a commodity too, and most of the time it behaves like one: it has a spot price, a forward curve, hedgers and speculators, all the machinery we have built up across this series. But a staple food carries one extra property that no barrel of oil has, and that property changes everything when supply gets tight.

That property is this: **food is not optional, and the people who can least afford a price spike are the ones who spend the largest share of their income on it.** A factory can substitute aluminum for copper, or run a shift less if energy gets expensive. A household cannot stop eating. In a rich country, food is maybe 10-15% of household spending and a doubling of the rice price is an annoyance. In a poor country, food can be 40-60% of a family's budget and the staple grain alone can be 20-30%. When the price of that grain doubles, it is not an inconvenience — it is the difference between eating and not eating, and historically it is the thing that puts crowds in the street and topples governments. Bread riots are one of the oldest forms of political unrest on record, from the French Revolution to the Arab Spring.

So a government looks at a spiking staple price very differently from how it looks at a spiking oil price. A spiking food price is not just an economic problem; it is a **regime-stability problem**. And that reframes the whole market. In ordinary commodities, when the price rises, the market is doing its job — rationing scarce supply and signaling producers to make more. In a staple food, when the price rises far enough and fast enough, governments stop trusting the market to do its job and *intervene*. The most common, fastest, and most damaging intervention is the export ban.

### What an export ban actually is

An **export ban** (or its softer cousins: an export tax, an export quota, a minimum export price, a licensing requirement) is a government order that restricts or stops the outbound sale of a commodity to the rest of the world. The logic is direct: if our farmers can sell rice abroad at \$650 a tonne but our own poor can only afford \$450, then every tonne that leaves the country makes domestic rice scarcer and dearer. So the government slams the door: rice stays home, the domestic price falls (or at least stops rising), and the urban poor can still afford to eat.

From inside the country, this is humane and rational. It is also, from the point of view of the *world* market, an act of economic violence — because it removes supply from a market that, as we are about to see, has almost no supply to spare. And it sets off a chain reaction. Hold the mechanism in mind: a local shock raises the home price; the government bans exports to protect home consumers; world supply shrinks; importers panic and over-buy; the world price spikes far more; and neighboring exporters, watching their own prices rise, slam *their* doors too. Each link makes the next worse. That is the cascade in the cover figure, and the rest of this post is an unpacking of why each arrow is so much bigger than it looks.

### The cost-of-carry spine still applies — until it doesn't

It is worth being precise about where this fits into the series spine. Everything we built about the forward curve, [contango and backwardation](/blog/trading/commodities/contango-vs-backwardation-what-the-shape-of-the-curve-means), and the [convenience yield](/blog/trading/commodities/convenience-yield-and-the-cost-of-carry-why-the-curve-has-a-shape) still describes a food market in normal times. When grain is plentiful, the curve is in contango (storage costs push later-dated prices above spot) and you can run a cash-and-carry trade. When grain is scarce, the convenience yield of holding the physical grain *right now* explodes, the curve flips into steep backwardation, and the spot price runs away from the deferred. That is the convenience yield doing exactly what we said it does: the value of having the physical thing in your hands during a shortage.

What an export ban does is take the convenience yield and put it on steroids — because the ban does not just make the physical scarce, it makes it *legally unobtainable* across a border. A buyer in the Philippines is not paying up for grain that is merely tight; they are paying up for grain that an exporting government has made *illegal to ship*. The price has to rise far enough to either ration the remaining free supply or to bribe some other exporter into selling. The convenience yield, in other words, stops being a smooth economic quantity and becomes a step function set by a political decision. That is the moment the financial machinery of the series meets its limit, and policy takes over.

## Why thin trade is the whole story

Here is the fact that everything hinges on, and it is the one most people get wrong. When you hear that rice "nearly tripled," your instinct is that the world must have lost a third of its rice. It did not. The world lost almost none of its rice. What the world lost was access to the small sliver of rice that is actually *for sale internationally*.

![Before and after comparison of world rice produced versus traded](/imgs/blogs/food-security-export-bans-and-when-governments-hoard-4.png)

The world grows around 520 million tonnes of milled rice a year. But only about **45-55 million tonnes — roughly 8-9% — ever crosses an international border.** The other ~91% is grown and eaten inside the same country. China grows and eats its own rice. India grows and eats most of its own (it exports the surplus). Indonesia, Bangladesh, Vietnam, Thailand — rice is the staple, it is grown locally, and the overwhelming majority of it never enters world trade at all. Rice is, in the jargon, a **thin** market: the traded volume is tiny relative to the produced volume.

Compare that to a deep market like crude oil, where the majority of production is traded internationally and a single producer cutting output by a few percent moves the world price by a manageable amount. (For how OPEC manages that, see [OPEC+ and the supply curve](/blog/trading/commodities/opec-plus-and-the-supply-curve-the-cartel-that-sets-the-floor).) In rice, the math is inverted. Because the traded slice is so thin, a small *production* change at one big exporter becomes an enormous *trade* change. If India produces ~135 million tonnes and decides to keep an extra 10 million tonnes at home, that 10 million tonnes is barely 7% of India's crop — but it is roughly **20% of all the rice the entire world has for sale.** A rounding error in production is a catastrophe in trade.

This is the multiplier at the heart of food-security panics, and it is worth making the arithmetic explicit because it is so counterintuitive.

#### Worked example: the trade-thinness multiplier

Suppose world rice production is **520 million tonnes** and world rice *trade* is **52 million tonnes** — a tidy 10% traded ratio. Now suppose the single largest exporter, India, accounts for about **17 million tonnes** of that 52 (roughly a third of all world trade), out of an Indian crop of about **135 million tonnes**.

India has a weak monsoon and its crop falls 5%. That is a production loss of \$0 in dramatic terms — about 6.75 million tonnes, a small dent in a 135-million-tonne harvest, and India still has more than enough to feed itself. But India's government, nervous about domestic prices, decides to *protect the home market* by halting exports of non-basmati white rice entirely.

What just happened to the world market? India's ~17 million tonnes of normal exports vanish. World trade falls from 52 to about **35 million tonnes** — a **33% collapse in available supply**, triggered by a 5% dip in *one country's harvest*. The multiplier here is roughly 17/52 over 6.75/135, or about **6.5x**: a 5% local production shock became a ~33% global trade shock. In a market where buyers are price-inelastic (you cannot just stop eating), a one-third cut in available supply does not move the price by one-third — it can double or triple it, because the remaining free rice has to be violently rationed by price. The lesson to carry away: in a thin market, *production* and *trade* are two different worlds, and a tremor in the first is an earthquake in the second.

That 6.5x multiplier is the entire reason food-security policy is so explosive. In a deep market it would not exist. In a thin market it is the dominant force, and it is exactly what played out in 2007-08, 2010-11, 2022, and 2023.

### The 5%-broken rice benchmark

One more piece of foundation before the case studies, because you will see it everywhere. When people quote "the rice price," they almost always mean **Thai or Vietnamese 5%-broken white rice**, expressed in US dollars per tonne. The "5% broken" refers to the milling quality: in the milling process some grains break, and a batch with no more than 5% broken grains is a standard, high-but-not-premium export grade. It is the rice-market equivalent of "WTI" or "Brent" for oil — a reference grade and delivery point that the whole market quotes against. Thailand publishes a 100%-Grade-B and a 5%-broken; Vietnam's Food Association publishes its own 5%-broken benchmark. When you read that "rice hit \$650," it is this benchmark, in USD per tonne, that is being quoted. Keep that unit in your head — USD per tonne of 5%-broken — because every number in this post lives in it.

### Hoarding has two faces: the state buffer and the panic grab

The word "hoard" in the title is doing real work, and it points at two distinct behaviors that both amplify a food spike. The first is *official* hoarding: a government's **strategic grain reserve**. Many large food-importing and food-producing nations hold a national buffer stock — China's grain reserves are the largest and most opaque in the world, India runs enormous buffer stocks through its Food Corporation of India, and most Asian rice economies hold some public stockpile. In normal times these reserves are a stabilizer: the state buys when the harvest is fat and the price is low, and sells when the harvest is thin and the price is high, smoothing the cycle. That is the buffer working *for* stability.

But the same reserve becomes a *destabilizer* the moment a government decides to *build* it during a tight market. If a big importer, frightened by a spiking price, decides to top up its strategic reserve right when supply is scarce, it adds a huge, price-insensitive buyer to a market that is already short — pouring fuel on the fire. Part of what made 2007-08 so violent was that the global rice buffer had been quietly drawn down over the preceding years (China and India had let reserves shrink), so the cushion that normally absorbs a shock was thin, and then, mid-panic, buyers scrambled to rebuild stocks at exactly the wrong time. A reserve is a shock absorber when it is full and being released; it is an accelerant when it is empty and being refilled.

The second face is *private* hoarding — the panic grab. When households, traders, and millers see a staple price rising and hear of bans, the rational individual response is to *buy and hold more than you need*, because you fear it will be dearer or unavailable tomorrow. Every household that buys two extra sacks of rice "just in case" is, collectively, a sudden surge of demand at the worst possible moment. This is the food-market version of a bank run: the behavior that is sensible for any one actor is catastrophic when everyone does it at once. Export bans and panic-buying feed each other — the ban proves the fear was justified, which triggers more hoarding, which justifies more bans. The cascade in the cover figure is, at bottom, a *coordination failure*: a stampede where everyone is individually rational and the crowd is collectively insane. Understanding that both the state buffer and the private grab can flip from stabilizing to destabilizing is the key to reading where a tightening market sits on the knife-edge between calm and panic.

## Case study: the 2007-08 rice crisis, a panic with no shortage

The 2007-08 episode is the purest demonstration of the cascade because, more than any other, it happened *without an underlying shortage*. It is the textbook case that a food crisis can be manufactured entirely out of policy and fear.

The backdrop in late 2007 was a wheat and broader grain price surge, driven by a poor global wheat harvest, a weak US dollar, high oil prices feeding into fertilizer and freight, and the new biofuel mandates diverting corn into ethanol. Rice itself was fine — stocks were adequate, production was normal. But the *fear* generated by the wheat spike was contagious, and rice was about to demonstrate why thin markets are dangerous.

The sequence:

- **October 2007:** India, worried about domestic food inflation, banned exports of non-basmati rice (it later set a high minimum export price, which amounts to the same thing). India was a major exporter; its exit alone tightened the world market sharply.
- **Late 2007 - early 2008:** Vietnam, the world's #2 exporter at the time, announced it would suspend new export contracts and cut its export quota to ensure domestic supply. Another large chunk of world trade gone.
- **Early 2008:** With the two big Asian exporters retreating, the world market thinned to almost nothing tradable. The Philippines — the largest importer — panicked and issued enormous import tenders, repeatedly, willing to pay almost any price to secure rice for its subsidized distribution program. Each failed or expensive tender broadcast desperation to the market.
- **Spring 2008:** Egypt, Cambodia, and others added their own export restrictions. The price went vertical.

The 5%-broken Thai benchmark went from around **\$335/tonne in late 2007 to a peak near \$1,000/tonne in late April / early May 2008** — roughly a tripling in about six months. And then, almost as fast, it collapsed: once it became clear there was no actual shortage, and once Japan signaled it would release some of its WTO-obligation rice stockpile (a release that, in the end, barely needed to happen), the panic broke and the price fell back toward \$600 and then lower over the following year. Riots had broken out in dozens of countries at the peak. There was never a global rice shortage.

The human cost is what makes this more than a market curiosity. The World Bank estimated that the 2007-08 food-price spike pushed tens of millions of people into poverty and food insecurity, concentrated in exactly the import-dependent, low-income countries with no buffer stocks of their own — West Africa, the Caribbean, parts of South Asia. Haiti's government collapsed amid rice riots; Bangladesh deployed troops to guard distribution; Egypt put the army to work baking bread. And the cruel irony is that almost none of these consumers were short of rice in any physical sense — they were priced out of a market that a handful of exporting governments had thinned to the point of dysfunction. The Japanese stockpile detail is the tell: the mere *announcement* that a large reserve might be released was enough to break the panic, because the panic was psychological, not physical. The grain to calm the market barely had to move; the *belief* that it could move was sufficient. That is the signature of a policy-and-fear spike rather than a true scarcity — it can be talked up, and it can be talked back down.

#### Worked example: a rice-price-spike P&L for a panicked importer

Put yourself in the shoes of the Philippine import agency in early 2008. You normally import about **2 million tonnes** of rice a year, and you budgeted at the late-2007 price of roughly **\$350/tonne**, so your planned annual rice bill was about **2,000,000 × \$350 = \$700 million**.

The panic hits. Your tenders clear not at \$350 but at an average of, say, **\$900/tonne** for the volume you secure during the spike. The arithmetic of your import bill:

- Budgeted: 2,000,000 tonnes × \$350 = **\$700 million**
- Actual at panic prices: 2,000,000 tonnes × \$900 = **\$1.8 billion**
- Overrun: **\$1.1 billion** — more than 2.5x the plan — for the *same physical quantity of rice*.

That \$1.1 billion is the cost of buying into a panic in a thin market. And here is the cruel second-order effect: your own desperate, repeated, high-priced tenders were *visible to the whole market* and helped push the price higher, which raised the cost of every subsequent tonne you bought. The lesson: in a thin staple market, a large importer's panic is partly self-inflicted — the act of buying frantically moves the very price you are trying to beat. The disciplined importer who buys steadily through the cycle, holds a strategic buffer stock, and refuses to chase the spike pays a fraction of what the panicker pays.

## Case study: 2010-11 wheat, the Russian ban, and the Arab Spring

If 2007-08 showed the cascade in rice, 2010-11 showed it in wheat — and showed that a food price spike can do more than cause riots. It can be a contributing factor in toppling regimes.

![Wheat price history with the 2022 war and bans spike marked](/imgs/blogs/food-security-export-bans-and-when-governments-hoard-2.png)

In the summer of 2010, Russia suffered a catastrophic heatwave and drought — the worst in a century — that destroyed roughly a third of its wheat crop and set vast areas of farmland and forest on fire. Russia had become one of the world's largest wheat exporters. On 5 August 2010, the Russian government announced a **ban on grain exports** to protect its domestic supply and keep bread affordable at home. The ban was a textbook export-ban cascade: a domestic shock (the drought) raised the home price, the government banned exports to protect consumers, and the world wheat price — already nervous from the drought news — spiked.

CBOT wheat, which had been trading around \$5/bushel earlier in 2010, surged toward \$8-9 by early 2011, and global food-price indices hit record highs. Now follow the chain into the real world. Many countries in the Middle East and North Africa are the world's *largest per-capita wheat importers* — Egypt is the single biggest wheat importer on earth, and bread (*aish*, which in Egyptian Arabic also means "life") is the literal staple, heavily subsidized by the government precisely because its price is politically radioactive. When the world wheat price doubled, the cost of those subsidies exploded, and the price of unsubsidized bread rose sharply for hundreds of millions of people who already spent much of their income on food.

The **Arab Spring** that began in late 2010 and swept through 2011 had many deep causes — autocracy, corruption, youth unemployment, a generation with no political voice. No serious analyst says wheat prices *caused* it. But food prices were a genuine accelerant: the timing of the uprisings tracked the global food-price spike closely, "bread" was a literal chant in the Egyptian protests, and several studies since have shown that food-price spikes are statistically associated with social unrest in import-dependent, low-income countries. A drought in Russia, converted by an export ban into a global wheat spike, transmitted into the bread price in Cairo and Tunis, and helped light a fuse. That is the export-ban cascade operating at the scale of geopolitics — and it is why [geopolitics and unscheduled shocks](/blog/trading/event-trading/geopolitics-elections-and-unscheduled-shocks) and food markets are so tightly wound together.

#### Worked example: the political-economy math of a ban

Why does a government choose the ban even though it costs farmers and damages its reputation as a supplier? Put real numbers on the trade-off it faces. Take Russia in 2010 with a stylized example.

Suppose Russia would normally export **20 million tonnes** of wheat at a world price that the ban itself helped push to **\$300/tonne**. The **export revenue forgone** by banning is roughly:

- 20,000,000 tonnes × \$300/tonne = **\$6 billion** in foreign-exchange earnings given up.

Now the other side of the ledger — what the ban *buys* domestically. Russia consumes, say, **35 million tonnes** of wheat at home. By keeping the export grain inside the country, the domestic price is held down — suppose the ban keeps the home price \$80/tonne *lower* than it would otherwise have been. The **domestic consumer surplus protected** is roughly:

- 35,000,000 tonnes × \$80/tonne = **\$2.8 billion** of purchasing power kept in citizens' pockets, concentrated among the bread-buying urban poor.

On paper, \$6 billion forgone exceeds \$2.8 billion protected — the ban looks like a bad economic deal. But the government is not optimizing dollars; it is optimizing *political survival*. The \$2.8 billion is concentrated on the politically dangerous group (urban consumers who riot), while the \$6 billion is diffuse (export revenue and farmer profits that do not riot). And the cost of *not* acting — bread riots, instability, a threat to the regime — is, to a government, effectively infinite. So an export ban is rarely the right call for national wealth, but it is almost always the right call for the people in power, which is exactly why bans are so common despite being economically destructive. You cannot predict bans with an economic model; you predict them with a *political* model.

That trade-off is worth drawing out as a decision, because the structure repeats in every food-security episode and seeing the two columns side by side is the fastest way to predict which way a government will jump.

![Policymaker dilemma decision diagram ban exports versus keep exporting](/imgs/blogs/food-security-export-bans-and-when-governments-hoard-7.png)

Read the two columns as a fork the government stands at the instant the staple spikes. Option A, ban exports, buys cheap domestic food and a calm street at the cost of export dollars, farmer profits, and the country's standing as a dependable supplier — plus the side effect of making the *world* price worse and inviting neighbors to copy the move. Option B, keep exporting, earns the foreign exchange and preserves the reputation that future deals depend on, but at the cost of letting domestic food get dearer and accepting the political risk that comes with it. In a calm market the textbook answer is Option B: trade is wealth-creating, bans are wealth-destroying, and a reliable supplier earns a premium over time. But the decision is never made in a calm market — it is made in a panic, where the home price is spiking and the regime is exposed. In that state, the diffuse, long-run, dollar-denominated benefits of Option B are politically invisible, and the concentrated, immediate, regime-threatening cost of *not* protecting consumers dominates. That asymmetry is why, when a staple really spikes, almost every exporting government in history has chosen Option A. The reputational cost is real but deferred; the riot risk is real and *now*; and politicians discount the future steeply. If you want to forecast a ban, do not ask what is best for the country — ask what is most dangerous for the people currently in power.

There is one more layer the diagram hints at: the **time-inconsistency** problem. A government that bans exports in a crisis teaches every importer in the world that it cannot be relied upon when the chips are down. That memory raises the risk premium buyers demand from that origin in *every future year*, which is a slow, invisible tax on the country's farmers forever. India, Russia, and Indonesia have all paid this reputational tax — long-term buyers now route around them or hold larger buffers against the next ban. But because the cost is spread across a decade and the benefit is concentrated in one terrible week, the ban still wins. A trader's takeaway from the dilemma: an exporter that has banned before is *more* likely to ban again, not less, because it has already shown it will pay the reputational price when its domestic politics demand it. Past bans are the single best predictor of future bans.

## Case study: 2022, war and a wave of bans all at once

The year 2022 stacked nearly every food-security risk on top of each other simultaneously, and the export-ban cascade ran in multiple commodities at once. It is the modern stress test of everything in this post.

The trigger was Russia's invasion of Ukraine in February 2022. Ukraine and Russia together were roughly **a quarter of world wheat exports** and a huge share of corn, sunflower oil, and barley. Ukraine's Black Sea ports — the arteries through which its grain reaches Egypt, Lebanon, North Africa, and Asia — were blockaded. Overnight, an enormous chunk of world grain trade was physically trapped behind a war. CBOT wheat spiked toward its all-time intraday high near **\$12.94/bushel** in March 2022, the mark on the figure above. Fertilizer prices (Russia is a top fertilizer exporter) spiked too, threatening *next* year's harvests everywhere.

And then the cascade kicked in on top of the war:

- **India's wheat ban (May 2022):** India had a record-looking wheat crop and had promised to "feed the world" amid the Ukraine crisis. Then a brutal March-April heatwave shriveled the harvest, domestic prices rose, and on 13 May 2022 India abruptly **banned wheat exports** — the exact cascade reflex, days after promising the opposite. The world wheat price jumped again on the news.
- **Indonesia's palm-oil ban (April-May 2022):** Indonesia, which produces over half the world's palm oil (the most-used cooking oil on earth), **banned palm-oil exports** in late April 2022 to control its own soaring domestic cooking-oil prices. Global vegetable-oil prices, already high because Ukraine's sunflower oil was trapped, spiked further. Indonesia reversed the ban within about three weeks when domestic warehouses overflowed and its own farmers were hurt — a perfect illustration of how bans backfire on the banning country.
- **The Black Sea Grain Initiative (July 2022):** Brokered by the UN and Turkey, this corridor allowed Ukrainian grain to ship out of three ports under inspection. It worked, grain flowed, and prices eased substantially through late 2022 — until Russia repeatedly threatened to and finally did exit the deal in July 2023, re-injecting volatility.

The lesson of 2022 is that the cascade is not specific to one grain. It is a *general reflex* that fires across wheat, rice, vegetable oils, and fertilizer whenever any staple gets tight. Dozens of countries imposed some form of food-export restriction in 2022 — the most since 2008. Each was locally rational; collectively they amplified the spike for everyone, especially the poorest import-dependent nations.

Two details from 2022 are worth keeping because they teach the limits of the cascade. First, India's wheat ban is the cleanest demonstration that **promises mean nothing against domestic politics**. The Indian government had publicly offered to "feed the world" through the Ukraine crisis — and then reversed itself within weeks the moment its own harvest disappointed and home prices rose. A government's stated export intentions are worthless as a forecast; only its *domestic* price and political situation predict its behavior. Second, Indonesia's three-week palm-oil ban is the cleanest demonstration that **bans punish the banning country, often fast**. By cutting off the export outlet, Indonesia caused domestic palm oil to glut: mills had nowhere to send their output, tanks filled, farmers could not sell fresh fruit bunches, and the domestic price of crude palm oil *collapsed* even as the world price spiked. The pain to Indonesia's own farmers and the threat to its export revenue forced a humiliating reversal in about three weeks. The cascade has a self-correcting tail — but only after it has already done its damage to the world price.

The Black Sea Grain Initiative is the counter-example that proves the rule. Where a ban *removes* supply and spikes the price, a credible corridor that *restores* supply does the opposite: once Ukrainian grain started flowing again under inspection in mid-2022, wheat gave back much of its war premium through late 2022. The market does not need the physical situation to be *good*; it needs the *flow* to be credible. That is why every headline about the corridor — extension, threat, withdrawal — moved the price: the trade was never about how much wheat existed, but about whether it could move.

## Case study: India's 2023 rice ban and the reshaping of the market

The most market-defining single policy of recent years, and the one most relevant to this series' Vietnam track, was **India's July 2023 ban on non-basmati white rice exports.** India is not just *an* exporter of rice; it is *the* exporter — by itself roughly **40% of all world rice trade**, more than the next several countries combined.

![Horizontal bar chart of the top rice-exporting countries](/imgs/blogs/food-security-export-bans-and-when-governments-hoard-5.png)

Look at that concentration. India exports more rice than Thailand, Vietnam, Pakistan, and the United States *combined* in a normal year. When a market is that thin globally and that concentrated in one supplier, that one supplier's policy *is* the market. So when India, facing an erratic monsoon, domestic food-price inflation, and an upcoming election, banned the export of non-basmati white rice on **20 July 2023**, it removed the single largest source of internationally traded rice from the market in one announcement.

The price reaction was immediate and severe. The Vietnamese 5%-broken benchmark, which had been sitting around \$500-540/tonne, vaulted toward **\$650/tonne** within weeks and stayed elevated into 2024, eventually printing multi-year highs around \$650-660.

![Vietnam 5 percent broken rice price with the 2023 India ban spike](/imgs/blogs/food-security-export-bans-and-when-governments-hoard-3.png)

That chart is the cascade in one line. The price was stable, India banned, the benchmark gapped up roughly 20-30% in a matter of weeks, held high through 2024 as the ban persisted and buyers scrambled for alternative origins, and then mean-reverted hard in late 2024 and 2025 once India eased the restrictions (it lifted the non-basmati white rice ban in September 2024). A single government's decision to protect its own consumers, in the thinnest of the major grain markets, drove a global price round-trip of hundreds of dollars a tonne.

### Who won: the substitution beneficiaries

When the world's dominant exporter pulls back, the buyers do not vanish — they have to eat. They go to the *next* available exporters: Thailand and, above all, **Vietnam**. This is the substitution trade, and it is the part of the story most relevant to the [Vietnam rice and the export-ban dynamics of Asia](/blog/trading/commodities/vietnamese-rice-and-the-export-ban-dynamics-of-asia) post in this series, which goes deep on the Vietnamese side.

![Vietnam rice exports rising as India restricted exports](/imgs/blogs/food-security-export-bans-and-when-governments-hoard-6.png)

Vietnam's rice exports, which had run around 6.2-6.3 million tonnes a year for years, jumped to about **8.1 million tonnes in 2023 and a record ~9.0 million tonnes in 2024** as global buyers who could no longer source Indian rice came knocking. Vietnam was selling more tonnes *and* at much higher prices — the export windfall was substantial. The same dynamic helped Thailand. The world's loss of Indian rice was, in a narrow commercial sense, Vietnam's and Thailand's gain.

#### Worked example: the substitution gain for Vietnam when India banned

Quantify Vietnam's windfall. Take two years:

- **Pre-ban (2022-ish baseline):** Vietnam exports about **7.1 million tonnes** at an average 5%-broken price of roughly **\$470/tonne**. Gross export value ≈ 7,100,000 × \$470 ≈ **\$3.34 billion**.
- **Post-ban (2024):** Vietnam exports a record **~9.0 million tonnes** at an average price closer to **\$600/tonne** (the elevated post-India-ban level). Gross export value ≈ 9,000,000 × \$600 ≈ **\$5.4 billion**.

The swing is roughly **\$2 billion of additional rice-export revenue** for Vietnam in a single year — about **60% more** than the baseline — driven almost entirely by a policy decision made in New Delhi. More tonnes (+27%) *times* a higher price (+28%) compounds into a large value gain. The mechanism is direct: in a thin, concentrated market, the dominant exporter's loss is a mechanical transfer of revenue to the next-cheapest origin — so when you see one big exporter ban, the immediate trade is to look at who picks up the displaced demand. That is why a food-security shock in India shows up as a tailwind in Vietnamese export earnings and, eventually, in the share prices of Vietnamese agricultural exporters.

## The toolbox of restriction: a ban is just the loudest option

It is tempting to treat "export ban" as a single switch, but governments have a graded toolbox, and reading staple-policy risk means knowing the whole ladder — because the softer tools are early warnings of the loud one. Listed from gentlest to most extreme:

- **Export tax or duty.** The government slaps a percentage levy on exports. This raises the cost of selling abroad, keeps some grain home, and — usefully for the state — *raises revenue* rather than forgoing it. Russia has used a sliding wheat-export tax as a permanent mechanism; it is the velvet-glove version of a ban.
- **Minimum export price (MEP).** The government forbids exporting below a floor price (India used a \$950/tonne MEP on basmati and a \$490 floor on parboiled rice in 2023-24). Set the floor high enough and it is a ban in everything but name — no buyer will pay it — but it lets the government claim it has "not banned" exports while achieving the same effect and capturing the high price on whatever still ships.
- **Export quota or licensing.** The government caps total export volume or requires a license for each shipment, which lets it meter the outflow and turn the tap down without closing it. Vietnam managed its 2008 and 2020 rice exports this way, suspending and re-issuing contracts.
- **Outright ban.** The loud option: no exports of the specified grade, full stop. India's July 2023 non-basmati white rice ban, Russia's 2010 grain ban, Indonesia's 2022 palm-oil ban.

The reason this ladder matters to a trader is that governments usually climb it in order, and the lower rungs are *leading indicators* of the top rung. A new export tax or a rising minimum export price is the market telling you the political pressure is building and a full ban is on the table if the domestic price keeps climbing. By the time the outright ban is announced, the price has already gapped; the edge is in reading the tax and the MEP as the early tremors. When India quietly raised the parboiled-rice export duty and floated MEPs in mid-2023, the full non-basmati ban that followed was not a bolt from the blue — it was the predictable next rung, and anyone watching the ladder saw it coming.

### The domino dynamic: why one ban becomes five

The single most important thing to internalize about the cascade is that bans are *contagious between exporters*, not just between an exporter and the world. When one big exporter bans, every *other* exporter watches its own domestic price rise (because world demand has just been concentrated onto it) — and now *that* exporter's government faces the same dilemma, with the same political incentives, and reaches for the same lever. The first ban manufactures the conditions for the second.

This is the domino dynamic, and it is what turns a single-country shock into a systemic crisis. In 2008 the dominoes fell as India → Vietnam → Egypt → Cambodia in a matter of months. In 2022, the war was the first shock, but then India (wheat), Indonesia (palm oil), and dozens of smaller players restricted in sequence, each responding rationally to the price its predecessor's ban had created. The dominoes also fall in reverse: when the lead exporter *lifts* its ban, the pressure on everyone else eases, and the whole structure unwinds — which is why ban-removals come in clusters too.

#### Worked example: how a 10% cut snowballs through the dominoes

Make the snowball concrete. Start with world rice trade of **52 million tonnes**, split India 17, Thailand 8.5, Vietnam 9, Pakistan 5, others 12.5.

**Round 1:** India bans, removing **17 million tonnes** — trade available to the world drops to **35 million tonnes**, a 33% cut. The price gaps up, say, +25%, from \$500 to \$625.

**Round 2:** Thailand and Vietnam now see their *own* domestic prices rising as the world piles onto them. Suppose each, nervous, trims exports by 10% to cool the home market — Thailand by 0.85 mt and Vietnam by 0.9 mt, removing another **~1.75 million tonnes**. Small in isolation, but it lands on a market that is *already* down to 35 mt and panicking, so it pushes the price up another, say, +10%, to \$690.

**Round 3:** Importers, watching two rounds of bans, panic-buy and top up reserves — adding demand exactly as supply shrinks. The price overshoots toward \$750+ before any new physical shortage materializes.

The arithmetic point: the first domino did the heavy lifting (a 33% supply cut), but the *follow-on* dominoes — each individually a modest 10% trim — compound on top of an already-stressed market and drive the overshoot. The whole-system move is far larger than any single country's action. That is why you never model a staple ban as a one-country event; you model it as the first tile in a row, and you watch the *second* and *third* exporters' domestic prices as the tell for whether the dominoes will keep falling.

## Common misconceptions

**"A food price spike means the world ran out of food."** Almost never. The 2007-08 rice crisis tripled the price with *no* aggregate shortage — global stocks were adequate the whole time. Food spikes are usually about the thin *traded* layer being disrupted by policy and panic, not about the world's granaries being empty. The aggregate and the traded slice are two different worlds; the price lives in the thin slice.

**"Export bans protect the country that imposes them."** They protect domestic consumers in the very short run, but they routinely backfire. Indonesia's 2022 palm-oil ban crashed *domestic* prices so far that its own farmers were hurt and warehouses overflowed, forcing a reversal in about three weeks. Bans also destroy a country's reputation as a reliable supplier, which costs future export business and investment. And by signaling panic, they can worsen the very hoarding they were meant to stop. The protection is real but narrow, short-lived, and expensive.

**"The futures market is where the food spike happens, so speculators caused it."** The CBOT and ICE futures reflect the spike, and speculators trade it, but the 2007-08 rice spike happened largely in the *physical* market through government tenders and bans, with the futures following. The driver of staple-food spikes is overwhelmingly physical and political — crops and policy — not paper. Speculators are the messenger far more than the cause, a point the [financialization debate](/blog/trading/commodities/the-forward-curve-the-most-important-chart-in-commodities) in this series unpacks more generally.

**"Big harvests make a country food-secure, so it won't ban."** Production and trade are decoupled. India in 2023 had plenty of rice for itself and still banned exports — the ban was about *domestic price inflation and an election*, not about running short. A country can be awash in grain and still hoard it, because the decision is political, keyed to the domestic price and the political calendar, not to the size of the harvest.

**"Once a ban is announced, the move is over."** The announcement is the gap, but the cascade has a tail: copycat bans from neighbors, panic-buying by importers, and the persistence of the ban (India's 2023 ban lasted 14 months) keep the price elevated long after day one. And the *removal* of the ban is itself a large, tradeable event — the 2024-25 round-trip lower happened when India eased. The policy has a beginning, a middle, and an end, and each is a separate move.

**"Free trade and the WTO prevent food-export bans."** They do not. World Trade Organization rules explicitly *permit* temporary export restrictions to prevent or relieve "critical shortages of foodstuffs" (GATT Article XI), with only weak notification requirements that are routinely ignored. There is no enforceable global discipline on food-export bans, which is precisely why they remain the default crisis tool: a government pays no trade-law penalty for slamming the gate. Repeated attempts to even exempt humanitarian (World Food Programme) purchases from export bans have struggled to get universal agreement. The legal architecture that disciplines tariffs and subsidies has a deliberate hole exactly where staple food is concerned — so do not assume any treaty will stop the next ban.

## How it shows up in real markets

Pull the threads together with the dates and numbers, because the pattern repeats with eerie consistency.

**The trigger is always a domestic price problem, not necessarily a global one.** 2007-08: wheat fear spilling into rice. 2010: Russian drought. 2022: war plus an Indian heatwave plus Indonesian cooking-oil inflation. 2023: Indian monsoon worry plus an election. In every case the *exporting* government acted to protect its *own* consumers, and the world market was collateral damage.

**The magnitude is set by thinness and concentration.** Rice spikes hardest (≈8-9% traded, India ≈40% of trade) — it nearly tripled in 2008. Wheat is deeper and more diversified across exporters (US, EU, Russia, Canada, Australia, Argentina), so its bans bite hard but rarely triple it. Vegetable oils are intermediate, with Indonesia and Malaysia dominating palm. The thinner and more concentrated the market, the more explosive the ban.

**The cascade includes copycats and panic-buyers.** No ban happens alone. 2008 had India, Vietnam, Egypt, Cambodia, and others; 2022 had dozens of countries restricting some food export — the most since 2008. And every importer that panic-tenders (the Philippines in 2008) adds fuel. The spike is the *sum* of the bans plus the panic, which is why it overshoots so far.

**The reversal is as sharp as the spike.** Once the panic breaks — Japan's stockpile signal in 2008, Indonesia's reversal in 2022, India's easing in 2024 — the price falls fast, because the underlying physical situation was rarely as dire as the panic implied. A staple spike caused by policy is, almost by definition, reversible when the policy reverses.

**The whole thing is a regime, not a constant.** This is the cross-asset lens. Correlations between a staple and the rest of the world flip during a food-security episode: rice decouples from the broader grain complex, exporters' currencies and equity markets catch a windfall, importers' fiscal positions deteriorate. The relationship between food and everything else is a *regime* that the ban switches on and off, which is the same point the [agriculture and softs](/blog/trading/cross-asset/agriculture-softs-the-food-and-fiber-markets) cross-asset post makes from the allocator's side.

## The takeaway: how a trader reads policy risk in staples

So how do you actually use this? The risk in staple foods is not a weather model you cannot beat and a crop you cannot forecast better than the USDA — for that machinery, the [WASDE and supply-shock](/blog/trading/commodities/weather-the-wasde-and-the-supply-shock-trading-the-report) post is the toolkit. The unique, tradeable risk in *staples* is **policy**, and policy is more legible than weather because it is made by humans responding to visible pressure. Here is the watch-list.

**1. Watch the exporters, not the global balance.** A staple spike comes from a big exporter, so map the concentration first. For rice: India is the whole ballgame (~40% of trade); watch India's monsoon, domestic rice CPI, and government statements obsessively, and watch Thailand and Vietnam as the substitution beneficiaries. For wheat: Russia, the EU, the US, Canada, Australia, Argentina, Ukraine — more diversified, so any single ban bites less. For palm oil: Indonesia and Malaysia. The lesson of every crisis is that you predict the spike by watching the *dominant exporter's domestic situation*, because that is where the ban will originate.

**2. Watch the stocks-to-use ratio.** The single best fundamental gauge of how close a market is to a panic is the **stocks-to-use ratio** — ending stocks divided by annual consumption, the buffer expressed as a fraction of a year's use. A high stocks-to-use (say 35%+ for rice) is a fat cushion; the market can absorb a shock without panicking. A low stocks-to-use (the low-to-mid 30s and falling, or worse) is dry tinder — a small spark sets off a fire because there is no buffer to dampen it. When stocks-to-use is falling and a major exporter has a bad crop, you are one announcement away from a cascade. The 2007-08 spike happened against a multi-year drawdown in rice stocks (driven partly by China and India quietly shrinking reserves); the cushion was thinner than the headlines suggested. Read the number this way: a 35% stocks-to-use ratio means the world is holding about 18 weeks of buffer; drop it to 20% and you are down to roughly 10 weeks, and at that point any disruption to the *flow* — a ban, a port closure, a drought scare — has to be rationed by price because there is no spare stock to lean on. The price sensitivity to a shock is roughly *inverse* to the buffer: thin cushion, violent price. A useful habit is to track each major staple's stocks-to-use trend in the monthly WASDE alongside the dominant exporter's crop news; when both arrows point the wrong way at once, the tinder is dry and you size your policy-risk premium up.

**3. Watch the political calendar.** Bans cluster around elections and moments of regime stress, because that is when a government most needs cheap bread. India's 2023 ban came ahead of state and national elections. Russia's 2010 ban came amid a national emergency. If a dominant exporter has a marginal crop *and* a politically sensitive moment, the probability of a ban jumps. You are forecasting a political decision, so use a political model: who is in power, how exposed are they to urban food inflation, and what is the calendar.

**4. Trade the substitution, not just the spike.** The cleanest expression of a ban is often *not* buying the spiking benchmark (which gaps before you can react) but positioning in the **beneficiaries**: the next-cheapest exporters who pick up displaced demand. When India banned in 2023, the trade was Vietnam and Thailand — their export volumes and earnings, their exporters' equities, their currencies. The dominant exporter's loss is a mechanical transfer to the runners-up, and that transfer is more durable and less crowded than chasing the headline price.

**5. Respect the reversal.** A policy-driven spike reverses when the policy reverses, and it reverses *hard* because the physical fundamentals were rarely as dire as the panic. Buying a staple at the peak of a ban-driven spike is buying into the most reversible kind of move there is. The mature play is to fade the panic, hold a buffer stock if you are a real consumer, and be ready to position for the round-trip lower when the ban eases — as the Vietnamese 5%-broken price did through 2024-25.

Step back to the series spine. We said all along that a commodity price is a physical thing forced through a financial contract, and that the curve, storage, and convenience yield are the gears. Food is the place where a *sixth* gear appears that none of the financial machinery can price: the moment a government decides that feeding its own people overrides the contract, the curve, and the carry. The forward curve will tell you the cost of storage; it will never tell you when New Delhi or Moscow will close the export gate. That is the unhedgeable, unpriceable risk at the heart of staples — and the reason the cleverest thing you can do in a food market is not to model the crop, but to read the politics of the people who control the gate. Watch the exporters, watch the buffer, watch the calendar — and remember that in the thinnest markets on earth, the most violent prices come not from the world running out, but from everyone trying to keep their own share.

## Further reading & cross-links

- [Grains: corn, wheat, and soybeans, the calories that trade](/blog/trading/commodities/grains-corn-wheat-and-soybeans-the-calories-that-trade) — the grain complex, the growing-season calendar, and why a Midwest drought is a global price event.
- [Vietnamese rice and the export-ban dynamics of Asia](/blog/trading/commodities/vietnamese-rice-and-the-export-ban-dynamics-of-asia) — the Vietnam-side deep dive on the 5%-broken benchmark and how India's 2023 ban reshaped Asian rice trade.
- [Weather, the WASDE, and the supply shock: trading the report](/blog/trading/commodities/weather-the-wasde-and-the-supply-shock-trading-the-report) — how a USDA report moves a grain market in seconds, ending stocks, and the stocks-to-use ratio.
- [Geopolitics, elections, and unscheduled shocks](/blog/trading/event-trading/geopolitics-elections-and-unscheduled-shocks) — how to react when a war, an embargo, or a sudden policy decision reprices a market with no schedule.
- [Agriculture and softs: the food and fiber markets](/blog/trading/cross-asset/agriculture-softs-the-food-and-fiber-markets) — the allocator's view of agricultural commodities as a portfolio sleeve, and why their correlations are regimes.
