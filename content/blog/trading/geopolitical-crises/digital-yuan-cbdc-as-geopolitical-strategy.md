---
title: "Digital Yuan and CBDC as Geopolitical Strategy: China's Play to Reshape Global Finance"
date: "2026-06-28"
publishDate: "2026-06-28"
description: "The digital yuan is not just a payment upgrade — it is China's attempt to build a dollar-independent financial system, gain surveillance reach, and settle trade in renminbi without touching SWIFT."
tags: ["geopolitics", "cbdc", "digital-yuan", "china", "de-dollarization", "swift", "renminbi", "monetary-policy", "financial-warfare", "currency"]
category: "trading"
subcategory: "Geopolitical Crises"
author: "Hiep Tran"
featured: true
readTime: 43
---

When most people hear "digital yuan," they picture a slicker version of WeChat Pay — a phone app for buying noodles. That framing misses the point so completely that it is worth pausing on. The digital yuan, known officially as e-CNY, is the most ambitious state-backed money project of the last fifty years, and the reason it exists has very little to do with buying noodles. It exists because the People's Bank of China looked at the architecture of the global financial system, saw that almost every cross-border dollar payment on earth routes through plumbing that the United States can read and switch off, and decided to build an alternative set of pipes.

> [!important]
> The digital yuan is a geopolitical instrument wearing the costume of a payment app. Three goals sit behind it: (1) build settlement rails that do not depend on the dollar or SWIFT, so that trade — especially with sanctioned partners — can clear even when Washington objects; (2) give Beijing unprecedented surveillance and control over money flows, because programmable central-bank money can be tracked, expired, and geofenced; (3) slowly chip at the dollar's role as the world's reserve and invoicing currency. Progress on goals one and two is real and fast. Progress on goal three is slow and easily overstated. This post walks through all three with the actual numbers.

This is a long read because the topic is genuinely tangled. It sits at the intersection of monetary economics, payment-system engineering, sanctions law, and great-power politics, and the loudest takes on all sides tend to be wrong. The "dollar is finished" crowd points at every yuan oil deal as if the system were collapsing; the "nothing ever changes" crowd points at the yuan's tiny reserve share and declares the whole thing theater. Both are missing the mechanism. The interesting story is not whether the dollar survives — it will, for a long time — but how a determined state builds optionality around a system it cannot control, and what that optionality is worth.

Let me build this up from the ground.

## Foundations: what money rails actually are

Before we can talk about why a central bank digital currency matters geopolitically, we have to be honest about a thing most people never think about: when you "send money," nothing physical moves. No truck of cash drives from your bank to the recipient's bank. What moves is a set of ledger entries. Your bank debits your account and credits another bank; that other bank credits the recipient. Money is just a synchronized set of who-owes-what records, and "payment" is the act of updating those records in agreement.

Domestically this is easy, because every bank in a country ultimately holds an account at the same central bank. When Bank A pays Bank B inside China, both have accounts at the People's Bank of China (PBoC), and the PBoC simply moves reserves from one to the other. One ledger, one operator, done.

Cross-border is where it gets hard, because there is no single world central bank. If a bank in Shanghai needs to pay a bank in São Paulo, they do not share a common ledger. So the system improvised. Banks open accounts at each other — "correspondent" accounts — and chain them together. The Shanghai bank holds dollars at a New York bank; the New York bank holds a relationship with a bank in Brazil; the chain gets walked one hop at a time. And the messages that coordinate all this — "please debit my account and credit theirs, value date tomorrow" — travel over a messaging network called SWIFT, the Society for Worldwide Interbank Financial Telecommunication.

Here is the crucial, under-appreciated fact: **SWIFT is not where money lives; it is where instructions about money travel.** SWIFT is a messaging cooperative based in Belgium. It does not hold balances. But because almost all cross-border payment instructions ride on it, being cut off from SWIFT is like being cut off from the postal service that everyone else uses to send checks. You may still have money; you just cannot easily tell anyone to move it.

And because the dominant correspondent currency is the dollar, the actual settlement of most cross-border payments touches a US bank at some point. That is the choke point. The United States can compel its own banks — and, through secondary sanctions, foreign banks that want continued dollar access — to refuse to process payments for a target. When Iran or Russia gets "cut off from the financial system," this is the mechanism: SWIFT disconnection plus dollar-correspondent denial. The target's money does not vanish; its ability to instruct the world to move it does.

Now the geopolitical logic of e-CNY snaps into focus. If China can build a payment channel that (a) does not send instructions over SWIFT and (b) does not settle through a dollar correspondent in New York, then a whole class of US leverage simply does not apply to payments on that channel. That is the prize. Not noodles.

![Diagram comparing the e-CNY payment path to the traditional dollar and SWIFT path](/imgs/blogs/digital-yuan-cbdc-as-geopolitical-strategy-1.png)

## What a CBDC actually is, and what e-CNY is specifically

A central bank digital currency (CBDC) is digital money that is a direct liability of the central bank, held by the public. That phrase "direct liability of the central bank" is doing enormous work, so let me unpack it against the money you already use.

The cash in your wallet is a central-bank liability — a banknote is literally an IOU from the central bank, and that is why it is risk-free. But the money in your bank account is not. It is a liability of your commercial bank. If the bank fails, your deposit is at risk (which is why deposit insurance exists). When you tap a card or use WeChat Pay, you are moving commercial-bank money, with the central bank only settling the net positions between banks at the end of the day.

A retail CBDC collapses that distinction. e-CNY held in your wallet is a claim on the PBoC directly, exactly like a banknote, but digital. It does not sit on a commercial bank's balance sheet. This is a genuinely new thing for the public to hold, and it is why central banks worldwide are simultaneously fascinated and nervous: it changes the relationship between the central bank, commercial banks, and citizens.

China chose a **two-tier architecture** to manage that disruption. The PBoC issues e-CNY and runs the core ledger, but it does not deal with you directly. Instead, commercial banks and licensed operators distribute e-CNY wallets to the public, handle onboarding, and run the apps. The user-facing layer is the familiar one — you might hold e-CNY inside WeChat Pay, Alipay, or the dedicated e-CNY app — while the value itself is central-bank money passing through, not bank deposits.

![Stacked diagram of the three layers of the e-CNY system from central bank to apps](/imgs/blogs/digital-yuan-cbdc-as-geopolitical-strategy-5.png)

The two-tier design is politically clever. It keeps commercial banks in the loop (so they are not cut out and turned into enemies of the project) while giving the PBoC something it has never had at retail scale: a real-time, account-level view of money in circulation, and the technical ability to attach rules to individual units of currency. That second capability — programmability — is where the surveillance story lives, and we will come back to it.

For now, hold three facts: e-CNY is central-bank money the public can hold; it runs on a two-tier model so banks stay involved; and the PBoC keeps the master ledger. Everything geopolitical flows from that last point.

## How we got here: the timeline that explains the urgency

China did not wake up one morning and decide to digitize the yuan. The project has a history, and the inflection points tell you what Beijing was actually reacting to.

![Timeline of e-CNY development from 2014 research through 2025 international pilots](/imgs/blogs/digital-yuan-cbdc-as-geopolitical-strategy-3.png)

The PBoC stood up a digital currency research group around 2014, when this was a quiet, almost academic effort. The mood changed sharply in 2019, and the trigger was external: Facebook announced Libra, a proposed global stablecoin backed by a basket of currencies. To Beijing, a privately issued, dollar-heavy, globally distributed digital currency run by an American tech giant looked like a sovereignty threat — a way for the dollar to colonize digital payments worldwide before China had a foothold. The e-CNY effort visibly accelerated within months.

Domestic pilots launched in 2020 in cities like Shenzhen and Suzhou, often via lotteries that handed residents small amounts of e-CNY to spend at participating merchants. The 2022 Beijing Winter Olympics were the first international showcase: foreign athletes and journalists could obtain and spend e-CNY, a deliberate signal that this was not a closed domestic toy. From 2023 onward, the focus shifted to the cross-border dimension — wholesale pilots, and live tests on a multi-CBDC settlement platform we will discuss shortly.

By the mid-2020s, the headline numbers were large in absolute terms and small in relative terms, a tension that runs through this entire subject. Cumulative e-CNY transaction volume had reached into the hundreds of billions of dollars, and well over 200 million wallets had been opened. Those are big numbers. But set against China's total retail payments — dominated by the entrenched WeChat Pay and Alipay duopoly — e-CNY remained a rounding error domestically. The domestic adoption has been, frankly, underwhelming. People already have payment apps that work perfectly; a third one with no obvious advantage and a clear surveillance angle is a hard sell.

This is the first thing to internalize: **e-CNY's domestic story is a disappointment, but its geopolitical story is the one that was always the point.** The cross-border rails, not the consumer app, are where the strategy lives.

## The de-dollarization question, with the numbers honestly stated

"De-dollarization" is one of the most abused phrases in financial commentary. Let me give you the real picture, because the data simultaneously supports and undercuts the breathless headlines.

Start with reserves. Central banks around the world hold foreign-exchange reserves, and the share held in dollars is a clean proxy for the dollar's dominance as a store of value. That share has genuinely declined over the last two decades.

![Line chart of US dollar versus Chinese yuan share of global FX reserves from 2000 to 2025](/imgs/blogs/digital-yuan-cbdc-as-geopolitical-strategy-2.png)

In 2000, the dollar was about 71% of allocated reserves. By 2025 it had drifted down to roughly 58%. That is a meaningful slide — about thirteen percentage points over twenty-five years. The "dollar is declining" camp points at this and is, narrowly, correct.

But look at where the lost share went. It did **not** go to the yuan. The yuan's share of global reserves rose from essentially nothing to a peak around 2.8% in 2021, then actually *fell back* to roughly 2.1% by 2025. The dollar's lost ground was picked up not by a single rival but by a scatter of "nontraditional" reserve currencies — the Australian and Canadian dollars, the Swedish krona, the Korean won, and gold. The world is diversifying *away* from the dollar at the margin, but it is not diversifying *into* the yuan. It is diversifying into a basket.

This distinction matters enormously for the e-CNY thesis. China can build the best cross-border rails in the world, but reserve status is not granted by good plumbing. It is granted by trust: trust that you can move your money out freely (China maintains capital controls), trust in the rule of law and independent courts (China's are subordinate to the Party), and trust that the asset will be there when you need it. Reserve managers diversifying away from the dollar are looking for *more* safety and liquidity, and the yuan offers *less* on both counts. A faster payment app does not fix a capital-control problem.

So when you read that "China is de-dollarizing trade," translate it carefully. China is reducing *its own* and *its partners'* dependence on the dollar for *transactions* — invoicing and settling trade in yuan. That is real and growing. But making the yuan a global *store of value* that the world wants to hold is a different and much harder goal, and on that front the needle has barely moved. e-CNY helps with the first and does almost nothing for the second.

#### Worked example: the USD selling-pressure thesis, sized

Let me put a number on a claim you will hear constantly: "if trade shifts to the yuan, the dollar will collapse from selling pressure." Is that mechanically plausible? Let's size it.

Global merchandise trade runs around \$25 trillion per year. Estimates of the share invoiced in dollars vary, but roughly 50% of global trade and the large majority of commodity trade is dollar-denominated; call it \$12.5 trillion of dollar-invoiced trade flow per year as a round figure.

Now suppose a dramatic shift: 5% of that dollar-invoiced trade converts to yuan invoicing over some period. That is \$12.5 trillion × 5% = \$0.625 trillion, or about \$625 billion of trade that no longer needs dollars to settle.

Does \$625 billion of trade flow translate into \$625 billion of dollar *selling*? No — and this is the error in the scary version. Trade flows are *gross* and largely *recycled*. A firm that earns dollars from exports spends them on imports or holds them briefly; the dollar is a medium of exchange that turns over many times. The relevant question for the dollar's *price* is the change in net *demand to hold* dollars, not the gross transaction volume passing through them.

Against a backdrop where global central banks hold roughly \$12 trillion in reserves and daily FX turnover exceeds \$7.5 trillion, a \$625 billion annual reduction in transactional dollar demand — spread over years, and partially offset by those dollars being held elsewhere — is a real but second-order effect. It would press on the dollar at the margin. It would not collapse it. The honest framing: de-dollarization of *trade* erodes a pillar of dollar demand slowly; it does not knock the building down. Anyone showing you a hockey-stick chart of yuan trade settlement and implying imminent dollar collapse is conflating gross flow with net demand.

## CIPS: the rail that matters more than the app

If you want to understand China's actual progress on dollar-independent settlement, stop watching the consumer e-CNY app and start watching CIPS — the Cross-Border Interbank Payment System. CIPS is China's alternative settlement and clearing system for cross-border yuan payments, launched in 2015. It is the wholesale plumbing, and it has grown a lot.

![Line chart of annual value processed by China's CIPS payment system from 2015 to 2024](/imgs/blogs/digital-yuan-cbdc-as-geopolitical-strategy-7.png)

CIPS started small and has scaled into the tens of trillions of yuan in annual processed value, with well over a thousand participating institutions spread across more than a hundred countries by the mid-2020s. Those participants are mostly indirect — they reach CIPS through a smaller set of direct participants — but the network reach is real.

Two clarifications that commentators routinely botch:

First, **CIPS is not (yet) a full SWIFT replacement.** SWIFT is a messaging network for *all* currencies; CIPS is a clearing-and-settlement system for the *yuan specifically*. Many CIPS transactions still use SWIFT for the messaging layer and only use CIPS for the yuan clearing. So CIPS reduces dependence on the dollar-correspondent leg without fully escaping SWIFT. That said, CIPS has its own messaging standard and can operate without SWIFT, and that optionality is precisely the strategic point — if SWIFT access were ever weaponized against China, CIPS provides a fallback that already exists and already works.

Second, **scale is relative.** Even at tens of trillions of yuan per year, CIPS processes a small fraction of what SWIFT-coordinated dollar settlement handles daily. CIPS is a credible, growing alternative rail, not a replacement. The strategic value is not that it has won; it is that it *exists and functions*, so the threat of cutting China off from dollar plumbing is far less terrifying than it would have been in 2010.

This is the recurring pattern in China's monetary statecraft: build the alternative *before* you need it, so that the West's financial weapons lose their deterrent edge. You do not have to use the gun for owning it to change the other side's behavior.

#### Worked example: SWIFT correspondent cost vs direct rail for a \$10,000 trade payment

Let me make the cost argument concrete, because cost is one of the legitimate non-coercive reasons partners might adopt yuan rails.

Imagine a small importer in a developing country paying a Chinese supplier \$10,000.

**Path A — traditional dollar correspondent banking.** The payment converts local currency to dollars, routes through one or more correspondent banks, then converts to yuan (or stays dollar and the supplier converts). Each hop takes a cut. For a corporate trade payment of this size in many emerging-market corridors, the all-in cost — FX spread plus correspondent fees plus the receiving bank's lifting fees — commonly runs 2% to 4%. Take the middle: 3%. On \$10,000 that is \$300 in friction, and settlement can take one to three business days.

**Path B — direct yuan settlement via CIPS, or e-CNY at the retail edge.** If the importer can source yuan locally (say through a yuan clearing bank in their region) and pay the supplier directly in yuan over CIPS, the dollar leg disappears entirely. There is still an FX spread to convert local currency to yuan, but only *one* conversion instead of round-tripping through the dollar, and no chain of correspondent lifting fees. Realistically this might cost 0.5% to 1.0%; call it 0.75%, or \$75 on \$10,000, with same-day or next-day settlement.

The saving is \$300 − \$75 = \$225 on a single \$10,000 payment, roughly a 2.25-percentage-point reduction in cost, plus faster settlement. Scale that across a country's entire import bill from China and the numbers get serious. **This is the quiet, legitimate engine of yuan adoption that has nothing to do with sanctions:** for trade *with China specifically*, paying in yuan can simply be cheaper and faster, because you stop renting the dollar's plumbing for a transaction that never needed the dollar in the first place. The geopolitical channel gets the headlines; the cost channel does much of the actual work.

## mBridge: the wholesale CBDC project built to skip the dollar

Now we reach the piece that should make Washington pay attention, and it is not the retail e-CNY app. It is mBridge — Project mBridge — a multi-CBDC platform incubated under the Bank for International Settlements that connects the central banks of mainland China, Hong Kong, Thailand, and the United Arab Emirates, with other participants and observers around the edges.

The premise of mBridge is to let commercial banks in different jurisdictions hold and exchange *wholesale* CBDCs of multiple countries on a shared platform, and settle cross-border payments *directly* between them in central-bank money — no correspondent chain, no dollar leg, and crucially no SWIFT message. Two banks in two countries can settle an international payment in seconds on a common ledger, with the transaction finality of central-bank money.

Read that again and sit with the implication. The entire architecture of dollar leverage — correspondent banking plus SWIFT messaging plus dollar settlement — is *bypassed by design* in mBridge. A payment between a Chinese bank and an Emirati bank on mBridge does not pass through New York, does not generate a SWIFT instruction the US can monitor, and settles with finality the US cannot reverse. For sanctioned or sanction-wary economies, that is the holy grail.

This is why mBridge is more strategically significant than the consumer e-CNY rollout will ever be. Retail e-CNY is about domestic control and a modest international showcase. mBridge is about rebuilding the wholesale cross-border settlement layer in a form the United States cannot see into or switch off. It is still early — volumes are pilot-scale, the participant set is small, and the BIS has at points stepped back from its central role amid the obvious geopolitical sensitivity — but the *capability* has been demonstrated to work. The proof of concept is no longer a concept.

#### Worked example: a \$50M oil purchase, mBridge vs SWIFT routing

Let me trace a concrete wholesale transaction both ways: a UAE bank settling a \$50 million oil purchase with a Chinese counterparty.

**SWIFT / dollar-correspondent routing.** The UAE bank sends a payment instruction over SWIFT. The transaction settles in dollars, which means it touches a US correspondent bank holding the dollar leg. Steps: UAE bank → its US correspondent → Chinese bank's US correspondent → Chinese bank. Settlement typically takes one to two business days. Costs include correspondent fees (a few basis points each, but on \$50 million even 5 bps is \$25,000) plus FX spreads if currency conversion is involved. And critically, **every step is visible to, and can be blocked by, US authorities**, because dollar clearing runs through US-supervised banks. If either counterparty were sanctioned, the payment dies at the correspondent.

Let's tally a plausible friction: assume 8 bps of total correspondent and processing cost on the dollar leg, which is \$50,000,000 × 0.0008 = \$40,000, plus one-to-two days of settlement lag (and the counterparty risk that comes with that lag).

**mBridge routing.** The UAE bank and the Chinese bank both hold the relevant wholesale CBDCs on the shared mBridge platform. The payment settles directly, central-bank money to central-bank money, in seconds. No US correspondent is in the chain. No SWIFT instruction is generated. The settlement is final on the shared ledger. Platform costs are minimal — the entire pitch is near-zero marginal settlement cost — call it a fraction of the correspondent path, plausibly under \$5,000 in all-in cost.

The cost saving (\$40,000 down to under \$5,000, so roughly \$35,000 on this single transaction) is meaningful but not the headline. The headline is the two things money cannot buy on the dollar rail: **invisibility to US oversight and immunity from US blocking.** For two counterparties who fear a future sanction — even counterparties who are not sanctioned *today* — that immunity is the product. They are not buying a cheaper payment; they are buying insurance against the dollar weapon. And once you understand that they are buying insurance, you understand why adoption can grow even when the cost saving alone would not justify the switching effort.

## Energy: where de-dollarization is most real

The single most important arena for yuan trade settlement is energy, and specifically oil and gas, because the "petrodollar" — the convention that oil is priced and settled in dollars — has been a load-bearing pillar of dollar dominance for half a century. If oil starts trading in yuan at scale, that pillar cracks.

China is the world's largest crude oil importer, which gives it enormous buyer's leverage. A buyer that large can credibly say to sellers: "price this in my currency, or sell to someone else." Beijing has been pressing exactly that argument, and it has had selective success. China has pursued yuan-denominated energy arrangements with Russia, Iran, and Saudi Arabia, and has notched symbolically important firsts — including a yuan-settled liquefied natural gas (LNG) cargo, where a European major sold LNG to a Chinese state buyer with the transaction settled in yuan rather than dollars. That a Western company settled an energy deal in yuan at all was the news; the volume was trivial.

The deepest case is Russia. After Russia's 2022 invasion of Ukraine triggered SWIFT disconnection for major Russian banks and a freeze of roughly half of Russia's central-bank reserves, Russia had no choice but to pivot its trade settlement away from the dollar and euro. China was the natural partner. By 2023, the overwhelming majority of Russia-China bilateral trade — figures around 90% are commonly cited — was being settled in rubles and yuan rather than dollars. The yuan became, in effect, Russia's external reserve and settlement currency, because it was the only large-economy currency available to a sanctioned state.

This is the clearest real-world demonstration of the entire e-CNY/CIPS/yuan strategy: when the dollar system is weaponized against a major economy, that economy *will* find an alternative, and China stands ready to be it. The sanctions did not isolate Russia from the global economy; they isolated Russia from the *dollar* economy and pushed it into the *yuan* economy. From Beijing's perspective, US sanctions policy has been the single most effective marketing campaign for yuan internationalization that money could not buy.

But — the honest caveat again — Russia is a special case born of necessity, not a model others are rushing to copy voluntarily. Saudi Arabia hedges and flirts with yuan oil deals to extract leverage from Washington and diversify, but the bulk of its oil still prices in dollars and its currency is still pegged to the dollar. The Gulf states want optionality, not a wholesale switch. The lesson is that the yuan grows fastest where the dollar is *withdrawn by sanction*, and grows slowly where it is merely *competed against*. That asymmetry shapes everything.

#### Worked example: sizing yuan-settled Russian oil exports to China

Let's estimate the scale of yuan-settled energy flow in the Russia-China corridor, because this is the largest concrete pool of de-dollarized trade.

China imports roughly 2 million barrels per day of crude from Russia (Russia having become one of China's top suppliers after 2022 as discounted Russian crude flowed east). Take an average realized price for that discounted Russian crude of around \$70 per barrel.

Annual crude value: 2,000,000 barrels/day × \$70/barrel × 365 days ≈ \$51 billion per year, just for crude oil, before adding pipeline gas, coal, and refined products which push total Russian energy exports to China well above that.

Now apply the settlement-currency mix. If roughly 90% of Russia-China trade settles in rubles and yuan, and the yuan is the dominant leg of that (Russia accumulates yuan it can actually spend on Chinese goods, whereas held rubles are less useful to China), then on the order of \$45 billion or more of *energy trade alone* is clearing outside the dollar in this single corridor each year. Add the non-energy trade — machinery, electronics, vehicles flowing from China to Russia — and total annual yuan/ruble-settled bilateral trade runs well over \$200 billion.

That is real de-dollarization, at material scale, in a major economy. It is also, importantly, *concentrated* — it is overwhelmingly one corridor (Russia-China) created by one event (the 2022 sanctions). The global picture is far more dollar-dominated than this corridor suggests. Both things are true: the Russia-China yuan corridor is genuinely large *and* it is the exception, not the emerging rule. Hold both facts at once or you will misread the whole subject.

## The surveillance dimension: programmable money

So far we have treated e-CNY mostly as a payments and geopolitics story. There is a second dimension that is, if anything, more consequential for the citizens who use it: control. Central-bank digital money is *programmable* in a way that cash and even commercial-bank deposits are not, and programmability is a polite word for control.

Consider what the PBoC can technically do with money it issues and whose ledger it keeps. It can attach an *expiry date* to a unit of currency, forcing it to be spent by a deadline (a tool floated as a stimulus mechanism — "spend this within thirty days or it disappears"). It can *geofence* money so it only works in certain places. It can restrict *what categories of goods* a given balance can buy. It can see, in principle at account-level granularity, who paid whom, when, and for what. Cash is anonymous and untraceable; a banknote does not report home. e-CNY, by design, can.

China's official position emphasizes "controllable anonymity" — small transactions enjoy some privacy from merchants and intermediaries, while the central bank retains visibility. But the architecture makes the state the ultimate observer and rule-setter. In a system where social-credit mechanisms already exist and capital controls are policy, programmable central-bank money is an extraordinarily powerful instrument of statecraft turned inward. It can enforce capital controls automatically (money simply will not move in prohibited ways), implement targeted stimulus, and — in the darker reading — financially constrain individuals or groups by editing what their money can do.

This is why CBDC design is genuinely contested in democracies, where the same technical capabilities raise alarms about state overreach, and why several Western central banks have moved cautiously, designing for privacy and intermediation specifically to *prevent* the central bank from having this kind of granular control. The technology is neutral; the governance around it is not. China's choice to give the state maximal visibility is a feature from Beijing's standpoint and a warning from a civil-liberties standpoint.

For our geopolitical lens, the surveillance angle matters in two ways. Domestically, it tightens the state's grip on capital flows, which *reinforces* the capital controls that, paradoxically, *prevent* the yuan from becoming a true reserve currency. And internationally, it makes some potential partners wary: a foreign government weighing yuan rails has to consider that the operator of those rails is a one-party state with deep surveillance capability and a record of using economic tools coercively. The same control that makes e-CNY attractive to Beijing makes it suspect to partners who do not want to trade dependence on Washington for dependence on Beijing.

## The global CBDC landscape: China is early, not alone

It would be a mistake to treat e-CNY as a uniquely Chinese phenomenon. Central banks almost everywhere are working on CBDCs, partly *because* China moved first and nobody wants to be left behind on the future of money.

![Bar chart of the number of countries by stage of central bank digital currency development in 2025](/imgs/blogs/digital-yuan-cbdc-as-geopolitical-strategy-4.png)

The landscape by the mid-2020s looked like this: dozens of countries researching CBDCs, dozens more in active development, a meaningful cohort running pilots, and only a small handful actually launched at retail scale (early launchers were small economies like the Bahamas with its Sand Dollar, Nigeria with the eNaira, and a few Caribbean projects, most of which saw weak adoption). The big economies — the United States, the euro area, the United Kingdom, Japan — have been deliberate and slow, researching and piloting wholesale and retail designs without committing.

The United States is the strategically interesting laggard. A US CBDC — a "digital dollar" — has been studied but is politically fraught, caught between concerns about surveillance (the same programmability worries, now pointed at Washington), the disruption to commercial banks, and a thriving private alternative: dollar-backed *stablecoins*. This last point is underrated. Privately issued dollar stablecoins have quietly become a powerful vector for dollar *digitization* without a government CBDC. They put a digital, globally transferable dollar in anyone's hands, and they extend dollar dominance into crypto rails. In a sense, the US does not urgently need a digital dollar because the private market has built dollar tokens that already circulate globally — the very thing China feared from Libra, now realized through stablecoins, but reinforcing the dollar rather than threatening it.

So the global contest is not simply "e-CNY versus the dollar." It is a three-way dynamic: state-issued CBDCs led by China, the slow-moving official Western CBDC efforts, and the explosive growth of private dollar stablecoins that entrench the dollar in digital form. China is ahead on the *state CBDC* track. The dollar is, somewhat accidentally, ahead on the *digital-circulation* track via stablecoins. These are different races, and conflating them produces bad analysis.

## Who actually adopts yuan rails, and why

Strip away the noise and the adoption logic is straightforward: a country leans toward yuan settlement rails to the degree that it (a) trades heavily with China, and (b) fears or has already suffered dollar-system exclusion. Plot those two axes and you can predict the geography of de-dollarization fairly well.

![Matrix of countries grouped by sanction exposure and trade ties to China](/imgs/blogs/digital-yuan-cbdc-as-geopolitical-strategy-6.png)

The strongest pull is on economies that are both sanction-exposed and China-dependent: Russia (the archetype), Iran, Venezuela. For them yuan rails are not a preference but a necessity, because the dollar system is closed or closing to them. The middle band is hedgers — Gulf states, Pakistan, parts of Africa and Latin America — who want optionality and cost savings without fully committing, using the *threat* of yuan adoption as leverage with Washington as much as actually switching. The third band is large China-trading economies with no sanction fear — Brazil and parts of Southeast Asia — who adopt yuan settlement opportunistically because it is cheaper for the China-specific slice of their trade, while keeping the dollar for everything else.

Notice what is absent from the high-adoption category: rich, rule-of-law democracies with deep capital markets. They have the least reason to switch, because they are not sanction targets, they trust dollar institutions, and they value the dollar's liquidity and legal protections. This is the structural ceiling on yuan internationalization. The yuan can capture the *coerced* and the *cost-sensitive* and the *hedgers*, but it cannot easily capture the *trusting and unconstrained*, who happen to hold most of the world's investable wealth. A currency that wins mainly among the sanctioned and the wary will grow, but it will not dethrone a currency that wins among the wealthy and the free. That is the asymmetry that caps the whole project.

## What this means for the dollar, realistically

Pull the threads together and a measured forecast emerges. The dollar's role is *fraying at the edges* in ways that are real, measurable, and driven substantially by US sanctions policy and by China's patient construction of alternative rails. But it is *not collapsing*, and the yuan is not the heir apparent.

The fraying is concrete. Trade settlement in yuan is growing, fastest in sanctioned corridors and China-heavy trade. Alternative rails — CIPS, mBridge — exist and function, eroding the deterrent power of dollar-system exclusion. The dollar's reserve share has slid from 71% to 58% over twenty-five years. Energy, the petrodollar pillar, has visible yuan cracks. These are not nothing.

The resilience is equally concrete. The dollar still dominates reserves, FX turnover, trade invoicing, and global debt issuance by wide margins. The yuan's reserve share has gone *nowhere* and even slipped, because capital controls and rule-of-law deficits make the yuan unattractive as a *store of value* no matter how good the *payment rails* get. Diversification away from the dollar is flowing into a basket of other safe currencies and gold, not concentrating in the yuan. And the dollar is quietly winning the digital-circulation race through stablecoins.

The single most important insight to carry away is the distinction between the dollar as a **medium of exchange** and the dollar as a **store of value**. China's strategy — e-CNY, CIPS, mBridge, yuan oil deals — is a sustained, partially successful assault on the dollar's role as a *medium of exchange* for trade and settlement. It is *not* an assault on the dollar's role as a *store of value*, and on that front China is structurally disarmed by its own capital controls and political system. You can build payment rails. You cannot, by building rails, manufacture the trust that makes the world want to *hold* your currency. The dollar's deepest moat is not its plumbing — which China is slowly circumventing — but the trust, liquidity, and legal protection behind it, which China cannot replicate without becoming a different kind of country.

#### Worked example: a reserve manager's actual decision

To make the store-of-value point unavoidably concrete, step into the shoes of a central-bank reserve manager in a neutral emerging economy with \$100 billion in reserves, deciding how much yuan to hold.

Today they might hold roughly \$58 billion in dollar assets (Treasuries mostly), \$20 billion in euros, and the rest scattered across yen, sterling, Canadian and Australian dollars, and gold. Their boss asks: "Should we shift \$10 billion into yuan assets?"

The manager runs the checklist. *Liquidity:* can I sell \$10 billion of yuan bonds in a crisis within a day without moving the market? The Chinese onshore bond market is large but access is gated and foreign participation is managed; the honest answer is "not as freely as Treasuries." *Convertibility:* if I need dollars urgently, can I convert yuan to dollars at scale on demand? China's capital controls mean *maybe, subject to policy* — and "subject to policy" is exactly what a reserve manager cannot tolerate in a crisis, because reserves exist *for* crises. *Legal protection:* if there is a dispute, do independent courts protect my claim? Less so than in New York or London. *Yield:* Chinese yields may be attractive, a genuine point in favor.

Weigh it: the yield pickup is real but the liquidity-and-convertibility risk is precisely the risk reserves are meant to avoid. So the manager allocates a *token* position — maybe \$2 billion, 2% — for diversification and to signal goodwill to a major trading partner, but not the \$10 billion, and certainly not a core allocation. Multiply this same cautious calculus across every reserve manager on earth and you get exactly the chart we saw: a yuan reserve share stuck around 2%, unable to break higher *no matter how good the payment rails become*, because the binding constraint is trust and convertibility, not plumbing. This single decision, repeated globally, is why the dollar's store-of-value crown is far safer than its medium-of-exchange dominance.

## Second-order effects and what to watch

If you are tracking this for investment or policy reasons, here is where the meaningful signal lives, separated from the noise.

**Watch mBridge participation and volume, not e-CNY wallet counts.** Retail wallet numbers are vanity metrics. The strategically loaded indicator is whether mBridge graduates from pilot to production, whether more central banks join, and whether wholesale settlement volume scales. That is the rail that actually bypasses the dollar.

**Watch the energy invoicing currency, especially in the Gulf.** A genuine inflection would be a major Gulf producer settling a *meaningful share* of oil in yuan as standard practice, not a symbolic cargo. As long as the Saudis keep the dollar peg and price oil in dollars, the petrodollar holds. A break there would be a real signal.

**Watch US sanctions policy, because it is the accelerant.** Counterintuitively, the fastest way to grow yuan internationalization is for the US to expand secondary sanctions and freeze more reserves, because each such action teaches more countries that dollar access is conditional on Washington's goodwill, and pushes them to build optionality. Every reserve freeze is an advertisement for diversification. The most powerful force pushing the world toward alternative rails is not Chinese ingenuity; it is American financial coercion convincing fence-sitters they need a backup plan.

**Watch the stablecoin regulatory regime in the US.** If the US embraces and regulates dollar stablecoins as instruments of dollar power — exporting digital dollars globally through private rails — it could entrench dollar dominance in the digital era even as the official digital-dollar debate stalls. This is the dollar's underappreciated counter-move.

**Discount the headlines that conflate flow with stock, and trade with reserves.** Most alarmist de-dollarization coverage commits one of two errors: treating gross trade flow as net dollar selling pressure, or treating growth in yuan *trade settlement* as growth in yuan *reserve status*. Once you can spot those two errors, you can read the entire field critically.

## Conclusion: optionality, not conquest

The right mental model for China's digital-yuan strategy is not *conquest* but *optionality*. Beijing is not, on any realistic timeline, going to replace the dollar as the world's reserve currency — its own capital controls and political system foreclose that. What Beijing *is* doing, methodically and with real success, is building a parallel set of financial rails that reduce its vulnerability to US financial coercion and give its trading partners a usable alternative when the dollar system is closed to them.

e-CNY is the retail face of this; CIPS is the wholesale clearing layer; mBridge is the cross-border settlement frontier that genuinely bypasses the dollar and SWIFT; yuan energy deals are the demonstration that the strategy works where the dollar has been withdrawn. Together they do not topple the dollar. They *insure China against the dollar*, and they extend a lifeline to anyone Washington cuts off.

The deepest irony is that the most effective driver of this whole project has been US sanctions policy itself. Each time the dollar is used as a weapon, more of the world concludes it needs a backup, and China is ready with one. The dollar's medium-of-exchange dominance is being slowly chipped away — partly by Chinese engineering, largely by American coercion teaching the world to hedge. Its store-of-value dominance, protected by trust and convertibility that China cannot replicate, remains formidable. Both of those statements are true simultaneously, and any analysis that asserts only one of them is selling you a story rather than describing the world.

## Sources & further reading

- International Monetary Fund, Currency Composition of Official Foreign Exchange Reserves (COFER) — quarterly data on reserve currency shares, including the long dollar decline and the yuan's small share.
- Bank for International Settlements — publications on Project mBridge and multi-CBDC arrangements for cross-border payments.
- Atlantic Council, Central Bank Digital Currency Tracker — country-by-country status of CBDC research, development, pilots, and launches.
- People's Bank of China — working group white papers on the progress of research and development of e-CNY.
- Cross-Border Interbank Payment System (CIPS) — annual reports on participants, country reach, and processed value.
- SWIFT RMB Tracker — periodic data on the yuan's share of global payments messaging.
- Academic and policy literature on de-dollarization, the petrodollar system, and the geopolitics of payment infrastructure (BIS, Peterson Institute, and central-bank research notes).
