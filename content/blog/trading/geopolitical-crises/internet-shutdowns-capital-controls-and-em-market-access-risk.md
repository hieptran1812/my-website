---
title: "Internet Shutdowns, Capital Controls, and Emerging Market Access Risk"
date: "2026-06-28"
publishDate: "2026-06-28"
description: "When governments cut the internet or freeze capital flows, markets reprice instantly. Here is how internet shutdowns move FX in unofficial markets, and why NDF contracts are the institutional hedge."
tags: ["geopolitics", "emerging-markets", "capital-controls", "internet-shutdown", "russia", "myanmar", "iran", "ndf", "market-access-risk", "currency-risk", "runet"]
category: "trading"
subcategory: "Geopolitical Crises"
author: "Hiep Tran"
featured: true
readTime: 43
---

> **TL;DR** — Internet shutdowns are not just a human-rights story. They are a market-access shock that reprices FX in parallel markets within 48 hours. Iran's November 2019 fuel-protest blackout (5 days, 38 million users cut) pushed the dollar black-market premium +15% against the official rate almost overnight. Myanmar's 2021 coup halted its stock exchange for 90 days and collapsed the kyat from 1,330 to 2,600 per dollar on the street (-49%). Russia's Sovereign Internet Law, signed in 2019, gave the Kremlin the infrastructure to isolate RuNet from the global internet — and in February 2022 Moscow paired that network kill-switch with mandatory 80% FX export conversion, a ban on foreign security purchases, and a \$10,000 cash withdrawal cap. The institutional solution — and the only liquid instrument that survives all four layers of access risk — is the Non-Deliverable Forward (NDF), an offshore USD-settled contract that circumvents every layer of government control. This post explains the mechanism, the math, and the playbook.

## Foundations: How Market Access Becomes a Geopolitical Weapon

Markets are information systems. Prices work because buyers and sellers can communicate: submit orders, receive confirmations, transfer ownership, and move cash across borders. Every one of those steps requires infrastructure — telecommunications, banking rails, legal enforceability of contracts — and every one of those layers can be cut by a sovereign government.

This is not a theoretical risk. It is a documented, repeatable event with measurable market consequences. The Access Now KeepItOn coalition tracked 187 internet shutdowns globally in 2022, a record high. The countries most active in deploying network blackouts — India (84 shutdowns in 2022, mostly in Kashmir), Iran, Russia, Ethiopia, Cuba, and Myanmar — share a common feature: they also have active currency controls or capital account restrictions. That overlap is not coincidental. Internet shutdowns and capital controls are complements in the authoritarian toolkit. You shut the information channel at the same time you lock the money channel, because if citizens can still see the dollar rate on a foreign server, they will route around your official fiction.

![Shutdown-to-Market Transmission Chain showing political trigger through NDF spread spike](/imgs/blogs/internet-shutdowns-capital-controls-and-em-market-access-risk-1.png)

The figure above shows the full transmission chain. A political trigger — a protest, a coup, a war — leads within hours to an internet cut. Within one to two days, capital flight signals emerge in informal money markets: hawala networks, border cash exchanges, black-market currency desks. These informal pricing signals produce the FX black-market premium, which itself feeds into the NDF market as offshore participants price in the access risk. Finally, over a period of weeks, index providers like MSCI and FTSE Russell assess whether the country's market meets their access criteria — and can downgrade or remove it entirely.

Understanding this chain is the foundation. Once you see the sequence, you can anticipate where the dislocations will appear, which instruments will price the risk, and how to position.

### What Is a Capital Control?

A capital control is a legal restriction on the movement of money across a country's borders. Controls exist on a spectrum from mild (reporting requirements above a threshold) to severe (mandatory conversion of export earnings, bans on buying foreign securities, closure of foreign-currency bank accounts). Most EM countries have some form of capital control on the books even during peaceful periods; what changes during a crisis is how tightly those controls are enforced and how many new restrictions are layered on.

Controls serve two purposes from the government's perspective. First, they slow the depletion of foreign-exchange reserves — if citizens cannot legally buy dollars, the central bank does not have to sell them to maintain the peg. Second, they prevent the domestic price level from instantly adjusting to reflect the government's true creditworthiness. A country with 200% inflation and a fixed official exchange rate can sustain that fiction so long as no one is allowed to trade at the market-clearing rate. Internet shutdowns reinforce this by making it harder for citizens to see what the real rate is.

### What Is a Non-Deliverable Forward?

An NDF is an offshore FX forward contract settled in a hard currency (overwhelmingly USD) rather than the restricted local currency. You agree today on a forward rate — say, CNY 7.40 per USD in 90 days. At maturity, whoever was wrong pays the winner the net USD difference. No renminbi ever changes hands. No onshore bank account, no PBOC approval, no capital control restriction applies.

The BIS Triennial Survey 2022 put the daily global NDF market at approximately \$250 billion in notional, with the most actively traded currencies being INR, BRL, IDR, KRW, TWD, and CNY. For currencies under severe access risk — the rial, the kyat, the ruble during sanctions — the NDF is frequently the only liquid price-discovery mechanism available to international participants.

The spread between the NDF-implied rate and the official onshore rate is what traders call the **political VPN premium**: the market's price for access risk. When Iran goes dark, that spread explodes. When Myanmar's coup began, the NDF (to the extent one existed for the kyat) would have reflected the black-market rate rather than the central bank's fiction. The premium is not irrational; it is an accurate measure of the real cost of inaccessibility.

## The Political Moves

To understand how market access weaponization works in practice, it is worth examining the four cases in detail before turning to the financial mechanics.

### Iran: The Most Practiced Shutdown Operator

Iran has shut down the internet more frequently, and more precisely, than any other state with a significant domestic financial market. The most instructive event is November 2019. On November 15, the government announced a sudden 50-300% increase in fuel prices, triggering nationwide protests within hours. Within 24 hours, the authorities deployed what became known as SIAM — the Iranian System for Managing and Controlling the Internet — to execute a near-complete national blackout. For five days, roughly 38 million users were cut from the global internet. Domestic intranet services (banks, government services) remained operational on the national network, but access to global markets, currency information, foreign news, and VPN services was effectively blocked.

The financial response was immediate. Iran's currency, the rial, had been in multi-year decline since 2018 US sanctions reimposition, with the official rate increasingly divergent from the parallel market. During the 48 hours after the shutdown, the dollar black-market rate jumped approximately 15% against the official rate. The mechanism is straightforward: when information channels are cut, citizens who want to convert savings to hard currency cannot compare prices, cannot reach informal currency dealers via encrypted messaging, and cannot see where the rate is. The resulting panic — buy dollars at any price before prices become truly unknowable — produced the spike. VPN traffic, measured by Cloudflare, surged over 3,000% as citizens attempted to circumvent the shutdown.

Iran has repeated this pattern in subsequent protest cycles (September 2022, Mahsa Amini protests), each time pairing network shutdown with attempts to freeze informal currency market pricing. The pattern is now sufficiently established that institutional traders watching Iran monitor Netblocks (a real-time internet shutdown tracker) as a leading indicator for the parallel-rate move.

### Russia: The Sovereign Internet Law and RuNet Isolation

Russia's approach is architecturally distinct from Iran's. Rather than deploying emergency kill switches ad hoc, Russia built the legal and technical infrastructure for network isolation in advance. The Sovereign Internet Law (Federal Law No. 90-FZ), signed by President Putin on May 1, 2019, required all Russian ISPs to install deep packet inspection (DPI) hardware at internet exchange points, reroute all traffic through state-controlled exchanges, and demonstrate the ability to isolate the Russian network (RuNet) from the global internet entirely.

Russia conducted a series of public isolation tests in 2019 and 2021, simulating what a fully segmented RuNet would look like. The stated justification was cybersecurity — protecting Russian infrastructure from a hypothetical Western cyberattack that severed Russia from global internet routing. The more accurate interpretation is that it gave Roskomnadzor (the state media and telecom regulator) an on/off switch for Russian participation in global information markets.

When Russia invaded Ukraine on February 24, 2022, the RuNet infrastructure was not used to create a full blackout — Russia's population of 144 million was too internet-dependent for a complete cut to be politically manageable internally. Instead, the authorities used DPI-based selective blocking. Twitter was slowed to unusability (throttled, not blocked) within days of the invasion. Instagram was fully blocked within two weeks, with over 150,000 blocking orders issued by Roskomnadzor in 2022 alone. International financial data terminals were progressively restricted.

What made Russia unique was the combination of network restriction with the most comprehensive set of hard capital controls deployed by a G20 country in the modern era. On February 28, 2022, Russia announced: mandatory conversion of 80% of foreign currency export earnings within three days; a ban on Russian residents transferring foreign currency abroad; a prohibition on foreign investors selling Russian securities (later moderated); a cap of \$10,000 on cash withdrawals from foreign-currency accounts; and suspension of short sales in equities. The ruble fell from 79.3 per dollar before the invasion to a peak of approximately 116 per dollar in early March 2022 — a 47% devaluation. By May 2022, the mandatory export conversion rule and oil revenue windfall had pushed the ruble back to approximately 67 per dollar, making it one of the fastest large-currency recoveries on record. The recovery was not a market phenomenon; it was the result of forced dollar supply from exporters.

![USD/TRY exchange rate 2015-2025 with capital control events annotated](/imgs/blogs/internet-shutdowns-capital-controls-and-em-market-access-risk-2.png)

The Turkish lira chart illustrates a different but related phenomenon: what happens when a country with significant capital account openness repeatedly cycles through currency crises. The 2018 peak of 7.24 TRY/USD, the 2021 collapse to 18.0, and the 2023 post-election jump to 25.4 each represent moments where the gap between the central bank's desired rate and the market-clearing rate became too large to sustain without intervention. Turkey is not a shutdown country — it is an access-control country, using reserve depletion and swap lines rather than network cuts. The pattern of instability driven by the tension between official and market rates is identical.

### Myanmar: The Coup as Full Market-Access Event

Myanmar's military coup of February 1, 2021 is the clearest modern example of a full market-access event triggered by a political discontinuity. At 3am local time, the military detained Aung San Suu Kyi and the elected civilian government, declared a year-long state of emergency, and within hours began disrupting mobile internet access. By February 6, mobile internet was largely shut down for the general population, with only fiber-connected businesses and government entities maintaining connectivity.

The Yangon Stock Exchange (YSX), Myanmar's three-year-old stock market, halted trading on February 1 and did not resume for 90 days. At the time of the halt, there were only a handful of listed companies with limited daily volume, but the symbolism was total: the only formal capital market in the country was closed.

The kyat, which had traded at approximately 1,330 per dollar in the days before the coup, rapidly devalued. The official Central Bank of Myanmar rate moved to approximately 1,800 per dollar over the following year — a 26% official devaluation. But the informal/black market rate reached 2,600 per dollar, representing a 49% devaluation from pre-coup levels and a 44% premium over the official rate at the same time. Foreign direct investment in Myanmar fell 60% in 2021. GDP contracted approximately 18%.

![Major internet shutdown events 2019-2023 with FX impact markers](/imgs/blogs/internet-shutdowns-capital-controls-and-em-market-access-risk-3.png)

### Ethiopia: The Invisible Shutdown

Ethiopia's Tigray conflict produced one of the longest and most complete internet shutdowns of the modern era: over 16 months of near-total communications blackout in the Tigray region, beginning in November 2020. The financial consequences for Ethiopia were less immediately legible in traded markets — the birr is not widely traded offshore, and Ethiopia does not have a significant capital market. But the broad dynamics were visible: informal dollar markets in Addis Ababa widened their spread against the official rate during the conflict period, the International Monetary Fund repeatedly flagged Ethiopia's parallel-market premium as a macroeconomic concern, and foreign-currency reserves fell to critically low levels as the war disrupted export revenues and informal capital flight accelerated.

The Ethiopia case illustrates a crucial point about internet shutdowns and market access risk: the magnitude of the financial impact depends heavily on whether the affected country is already integrated into global capital markets. A country with no NDF market, no foreign-equity investors, and no offshore bond market can sustain a shutdown with less immediate global financial reaction — but that does not mean there are no financial consequences. It means those consequences are expressed domestically (hyperinflation, parallel-market premiums, collapse of formal credit) rather than in offshore instruments.

## The Financial Channels

Having established the political context, we can now map the specific financial channels through which internet shutdowns and capital controls transmit into market prices. There are four primary channels.

![Country x Shutdown Type x Market Impact Matrix](/imgs/blogs/internet-shutdowns-capital-controls-and-em-market-access-risk-4.png)

### Channel 1: FX Black Market Premium

The most immediate and direct channel is the parallel currency market. In any country with formal capital controls, there exists an informal market where the local currency trades at a different rate than the official one. The premium of the black market rate over the official rate (measured in dollar terms, i.e., how many more local currency units are needed to buy one dollar informally versus officially) is the most direct measure of the market's distrust of the official rate.

During a multi-day internet shutdown in a country with currency controls, the following dynamics compound:

1. Demand for hard currency increases as citizens seek safety and information control fails.
2. Supply of hard currency to informal markets is disrupted because dealers cannot communicate via encrypted messaging, cannot check global prices, and cannot safely transact.
3. The resulting demand-supply imbalance produces a sharp premium spike.
4. As the shutdown extends, the premium stabilizes at a higher level representing the new risk-adjusted equilibrium.

The historical data is consistent: every multi-day shutdown in a country with pre-existing currency controls (Iran November 2019, Myanmar February 2021, Cuba July 2021, Iran September 2022) was accompanied by a parallel-market premium spike of 10-50% within the first 48-72 hours.

### Channel 2: NDF Market Re-pricing

For countries with active NDF markets, the offshore price-discovery mechanism is the first place where institutional investors express access-risk repricing. The NDF does not require any onshore market access, onshore bank account, or government permission. Two counterparties — typically banks or funds in London, Singapore, or New York — agree on a forward rate settled in USD. The only inputs required are a public reference rate (usually a central bank fixing) and counterparty credit.

When market access risk rises — due to a coup, sanctions announcement, or shutdown signal — NDF traders reprice the forward rate to reflect:
- The probability that the official fixing rate will be moved by government decree
- The cost of capital tied up in a position that cannot be unwound onshore
- A liquidity premium for the reduced number of counterparties willing to take the other side
- A tail-risk premium for complete exclusion from the index (MSCI review)

For Iranian rial NDFs during periods of heightened sanctions and shutdown risk, this complex of premia has produced annualized NDF premiums of 400-600% above the official rate. That number sounds surreal until you account for the fact that the rial has lost 99% of its value against the dollar over the past decade, with most of that depreciation happening in discrete crisis events rather than smoothly.

### Channel 3: Equity Market Halt and Index Exclusion

When a country's stock exchange closes — as Myanmar's YSX did for 90 days — or when a government restricts foreign investors from selling their existing equity positions (as Russia did briefly in February 2022), index providers face a choice: keep the country in the index at its most recent prices (creating a fictional valuation) or remove it.

MSCI moved with exceptional speed on Russia. On March 2, 2022, MSCI announced a consultation to remove Russia from all MSCI indices. By March 9, Russia's weight in the MSCI Emerging Markets Index — which had stood at approximately 3.2% in January 2022, representing roughly \$150 billion in tracking capital — was set to zero. The removal became effective March 9, 2022, eight days after the announcement.

The market impact was concentrated in passive ETFs and index funds tracking MSCI EM. The forced selling — estimated at \$1-3 billion across the largest passive vehicles — had to happen into a market where Russia's own exchanges were restricting foreign selling. The resolution for most passive managers was to write their Russia holdings to zero rather than attempt to sell at a market-clearing price that did not exist.

### Channel 4: Banking and Payment Rail Disconnection

The fourth channel — SWIFT disconnection — represents the deepest layer of access denial. When Russia was partially disconnected from SWIFT in March 2022, the impact on RUB-denominated transactions was immediate: Russian banks could not send or receive international wire transfers through the standard messaging system. Oil and gas trades had to be restructured through alternative payment systems. Coupon payments on Russian Eurobonds became legally and operationally uncertain, triggering credit events on CDS contracts.

The \$300 billion in Russian central bank reserves frozen by Western governments (the largest sovereign reserve freeze in history) was the most aggressive expression of this channel. Those reserves — held in EUR, USD, GBP, and JPY at foreign central banks — became inaccessible to the Russian government, eliminating their primary tool for FX market intervention.

## Worked Examples with Full Market Math

### Worked Example: Myanmar Kyat Arbitrage

On January 31, 2021 — the day before the coup — the Myanmar kyat traded at 1,330 per USD at the official Central Bank of Myanmar rate. One week after the coup, the informal market was pricing the kyat at approximately 1,600 per USD. By mid-2021, informal markets reached 2,600 per USD.

**Computing the arbitrage spread at peak:**

- Official rate: 1,800 MMK/USD
- Black market rate: 2,600 MMK/USD
- Spread in percentage terms: (2,600 - 1,800) / 1,800 = **44.4% premium** on the informal market

This means someone holding 1,800,000 kyat could obtain \$1,000 at the official rate, but the same \$1,000 cost 2,600,000 kyat on the street. Conversely, a trader who had access to official-rate dollars could, in theory, sell them at the street rate and pocket a 44.4% immediate return.

**Why is this not a riskless arbitrage?**

The spread looks enormous, but collecting it requires:

1. **Access to official-rate FX allocation**: Only government-connected entities and licensed banks could buy USD at 1,800. Small businesses and individuals had no legal channel.
2. **Physical dollar delivery risk**: Informal currency dealers could not safely advertise or operate during internet shutdowns and military checkpoints. Finding a buyer for your official-rate dollars at the street rate required trust networks that the shutdown disrupted.
3. **Legal risk**: Currency arbitrage in Myanmar during the coup was explicitly criminalized. The military government threatened jail time for informal currency trading.
4. **Timing risk**: The spread was volatile. A military announcement or temporary exchange rate intervention could compress it rapidly, leaving anyone mid-trade with a large loss.
5. **Capital blockage**: Even if you profited in kyat terms, you could not convert the profits back to USD through any legal channel, trapping gains in a depreciating currency.

The "arbitrage" was therefore available only to insiders with military-connected FX access and the operational ability to clear informal trades, which is a de facto description of crony rent extraction rather than market arbitrage.

![Myanmar Kyat official versus black market rate 2021-2022](/imgs/blogs/internet-shutdowns-capital-controls-and-em-market-access-risk-7.png)

#### Worked example: NDF Hedge Cost on Iranian Sovereign Bonds

Suppose your fund holds \$10 million notional in Iranian sovereign bonds (in practice, these would be held through obscure offshore structures or captured in a frontier-markets fund). The bonds pay coupons in IRR at the official rate. You want to hedge the currency risk using NDFs, but the IRR NDF market at this moment reflects a 400% annualized premium over the official rate.

**Step 1: Compute the annualized forward rate**

Official spot: IRR 42,000 per USD (illustrative 2023 official rate)
NDF-implied forward: reflecting a 400% annualized depreciation
90-day NDF forward rate = 42,000 × (1 + 4.00 × 90/365) = 42,000 × (1 + 0.986) = 42,000 × 1.986 ≈ **83,400 IRR/USD**

This means the market expects the rial to weaken from 42,000 to 83,400 against the dollar over 90 days — nearly doubling in local currency terms.

**Step 2: Compute the hedge cost**

To hedge \$10M USD notional of bonds, you enter a 90-day NDF to sell IRR and receive USD at 83,400 IRR/USD.

Hedge cost (annualized) = 400% × \$10,000,000 = **\$40,000,000 per year**

On a quarterly basis, that is \$10,000,000 per quarter just to maintain the hedge. This is not a cost in the sense of a direct cash payment; it is the opportunity cost embedded in the NDF price. If the rial does not depreciate as fast as the NDF implies, the hedger pays more than necessary. If the rial depreciates faster, the hedge underperforms.

**Step 3: Break-even analysis**

The hedge is break-even if the rial depreciates by exactly 400% annualized — i.e., if the spot rate at NDF maturity equals the NDF forward price. If the coupon on the Iranian bonds (in USD equivalent, at the official rate) is less than 400% per year, the hedged position is net negative return. If the bond yield (in USD terms) is, say, 12% per year, then:

Net hedged return = 12% − 400% = **−388% per year**

The fund is massively underwater on a hedged basis. The only rational reason to hold hedged Iranian bonds in this scenario is if you believe the NDF premium overestimates future depreciation — i.e., you are implicitly taking a long position on the rial's stabilization. That is a political risk bet, not a fixed-income trade.

**The practical implication:** Most institutional funds do not hedge Iranian bonds because the hedge cost exceeds the yield by an order of magnitude. The position either has to be unhedged (pure currency risk), or the fund writes it down to near-zero and holds it as an OTC illiquid position marked to recovery value.

#### Worked example: Russia MSCI Removal — Forced Passive Selling

Russia's MSCI EM weight was approximately 3.2% in January 2022. The MSCI Emerging Markets Index at that time tracked roughly \$2.0 trillion in assets (across ETFs, mutual funds, and institutional mandates with EM as benchmark). Russia's weight therefore represented approximately:

0.032 × \$2,000,000,000,000 = **\$64 billion** in benchmark-driven tracking exposure

This is the theoretical maximum forced sale if all passive managers simultaneously rebalanced to zero Russia weight. In practice:

- Not all EM exposure is passively managed; active managers had already begun reducing Russia exposure before the formal MSCI announcement
- Some passive managers chose to write Russia holdings to zero (marking as illiquid) rather than attempting to sell into a market with restricted foreign selling
- The Moscow Exchange's suspension of foreign selling limited how much could actually be transacted

MSCI announced the removal on March 2, 2022, effective March 9. Eight days. During that window, the Moscow Exchange restricted trading and limited foreign investor selling. Most large passive ETF managers (BlackRock, Vanguard, State Street) disclosed in subsequent filings that they had marked Russian equities at zero rather than selling — effectively taking a total loss on their tracker positions.

**The forced selling math for ETFs:**

iShares MSCI Emerging Markets ETF (EEM) had approximately \$28 billion in AUM in early 2022. At 3.2% Russia weight, theoretical Russia exposure was:

0.032 × \$28,000,000,000 = **\$896 million** in Russian equities in EEM alone

With selling restricted by Moscow Exchange rules and most Russian ADRs (American Depositary Receipts, the US-traded version of Russian stocks) suspended by NYSE and NASDAQ, the practical outcome was that EEM's Russia position was written to zero. Shareholders bore the loss directly — the ETF's NAV fell by approximately 3% purely from the Russia write-down, independent of any price movement in other EM holdings.

**The lesson for index-weight risk:**

Any country that represents more than 1% of a major EM index creates significant forced-selling risk for passive vehicles if it is removed. The removal does not have to be preceded by a shutdown or sanctions; it can come from a MSCI market access review triggered by capital controls alone (as happened when MSCI put Pakistan on its watch list in 2022 for liquidity restrictions). The time between announcement and effective date — typically 5-30 days — is the window for trading.

![NDF mechanics before and after hedging an inaccessible onshore position](/imgs/blogs/internet-shutdowns-capital-controls-and-em-market-access-risk-5.png)

#### Worked example: Internet Shutdown FX Event Study

**Methodology:**

Using the Iran November 2019 and Myanmar February 2021 cases, we can construct a rough event-study framework for estimating the expected parallel-market FX move from a multi-day shutdown in a country with strict capital controls.

**Iran November 2019 (5-day shutdown):**

Pre-shutdown parallel rate: IRR ~130,000/USD (informal; official was ~42,000)
Post-shutdown parallel rate (peak during shutdown): ~150,000/USD
Implied move: +15.4% depreciation in the parallel rate over 5 days

**Myanmar February 2021 (ongoing partial shutdown):**

Pre-coup rate: 1,330 MMK/USD
Parallel rate 1 week post-coup: ~1,600 MMK/USD
Implied move in first week: +20.3% depreciation in the parallel rate

**Cuba July 2021 (protest-triggered shutdown, 2 days):**

Unofficial CUP/USD (black market): approximately 60 CUP/USD pre-protests
Post-shutdown: approximately 70-75 CUP/USD
Implied move: +17-25% depreciation

**Composite estimate:**

Across these three cases, a 2-7 day internet shutdown in a country with existing strict capital controls and a pre-existing parallel-rate premium of 50%+ appears to generate a parallel-rate depreciation of approximately **15-25% in the first 5 days** of the shutdown event. The move is front-loaded: most of the repricing happens in the first 48-72 hours as information uncertainty peaks.

**Critical caveat on causality:**

The shutdown is not causing the underlying economic deterioration — it is a symptom of the same political instability that is causing it. The shutdown happens because the government fears protests; the currency moves because the government is acting in ways that spook dollar demand. Disentangling the pure shutdown effect from the political-regime-change effect is difficult. The practical implication for traders is the same regardless of causality: when you see a multi-day shutdown activate in a country with capital controls, the parallel-rate move is likely in progress.

## How the NDF Market Functions as the Institutional Solution

![EM Market Access - the four-layer stack from onshore exchange to NDF offshore](/imgs/blogs/internet-shutdowns-capital-controls-and-em-market-access-risk-6.png)

The layered stack in the figure above is the key mental model for understanding why NDFs exist and why they are the institutional hedge of last resort. A government can cut access at any of the first four layers:

**Layer 1 — Onshore Exchange:** Close the stock exchange (Myanmar 90 days), halt FX trading (Russia partial), suspend specific securities. This requires regulatory action.

**Layer 2 — Capital Controls:** Restrict cross-border money flows, mandate export earnings conversion, prohibit foreign security purchases. This requires legal authority.

**Layer 3 — Internet and Information:** Deploy DPI-based blocking (Russia's Roskomnadzor), activate a kill switch (Iran's SIAM), throttle mobile internet (Myanmar). This requires technical infrastructure and is increasingly common.

**Layer 4 — External Sanctions and Reserve Freeze:** This is applied by foreign governments, not the country itself. SWIFT disconnection, central bank reserve freeze, asset sanctions on banks. This requires multilateral political coordination.

The NDF exists entirely outside all four layers. It is a bilateral OTC contract between two offshore counterparties, settled in USD through the standard correspondent-banking system, with no requirement for any connection to the country's domestic financial infrastructure. The only onshore input is a reference rate — typically a published central bank fixing rate or a polling composite. Even if that reference rate becomes fictional (because the government is maintaining a false peg), NDF counterparties can agree to use an alternative reference or simply close positions out at market-clearing prices.

### NDF Market Structure in Practice

The largest NDF markets by daily notional (BIS 2022 data):
- Indian Rupee (INR): ~\$50B/day offshore NDF
- Brazilian Real (BRL): ~\$35B/day
- Korean Won (KRW): ~\$30B/day (though Korea also has an onshore deliverable forward market)
- Chinese Renminbi (CNY/CNH): ~\$25B/day offshore NDF
- Indonesian Rupiah (IDR): ~\$20B/day
- Taiwanese Dollar (TWD): ~\$18B/day

For currencies under severe access risk — the Iranian rial, the Myanmar kyat — there is no liquid NDF market. The very illiquidity and state fragility that creates the access risk also prevents the development of an offshore hedging market. Investors in those currencies are either making a political bet on stabilization, or they are constrained by mandate to hold them and must accept the unhedged risk.

For mid-sized EM currencies (the Nigerian naira, the Pakistani rupee, the Egyptian pound), there are offshore NDF markets but with wide bid-ask spreads and limited depth. These markets function better as risk indicators than as practical hedging tools for large positions.

For major EM currencies (INR, BRL, KRW, CNY), the NDF market is deep enough to hedge significant positions. The political VPN premium in these currencies — the spread between NDF-implied and onshore spot — is typically tight (0.1-0.3%) in normal times and widens to 0.5-2.0% during acute political stress events.

### The Mechanics of NDF Settlement

When an NDF matures, the settlement is straightforward:

1. Compare the NDF forward rate (agreed at trade inception) to the fixing rate (published by the central bank or a polling composite on fixing date).
2. Calculate the net USD difference. If the forward rate was 7.40 CNY/USD and the fixing came in at 7.50 CNY/USD (weaker renminbi), the renminbi seller receives the difference (they were right about depreciation).
3. Wire the USD difference through the standard correspondent banking system.

No renminbi crosses a border. No Chinese bank account is needed. The trade is entirely offshore. This is what makes the NDF mechanism robust to all four layers of access control. The only scenario where an NDF cannot settle is if the fixing reference becomes completely unreliable — as might happen if a country suspended its central bank operations entirely. Even then, ISDA documentation provides fallback procedures.

## Common Misconceptions

**Misconception 1: "An internet shutdown only affects retail investors"**

Reality: Professional institutional investors rely on internet connectivity for order transmission, real-time data feeds, risk system updates, and counterparty communication. A complete shutdown does not just cut retail investors from their brokerage apps; it cuts the data connections that institutions' trading systems use to communicate with exchange matching engines. In practice, most institutional systems have redundant private network connections (dedicated leased lines, MPLS networks) that can survive a public internet shutdown. But for smaller funds operating through standard internet-connected systems, a complete blackout creates real operational risk.

**Misconception 2: "Capital controls prevent currency depreciation"**

Reality: Capital controls slow the speed of depreciation by preventing immediate mass conversion. They do not prevent depreciation if the underlying fundamentals — current account deficits, monetary expansion, foreign debt burden — are negative. Turkey's recurring lira crises demonstrate this: controls slow the move, but the longer the controls suppress the price, the larger the repricing shock when controls are eventually relaxed. Russia's 2022 experience is the extreme example: mandatory 80% export conversion produced a temporary ruble recovery, but by 2024 the ruble had depreciated well beyond its pre-war level as the underlying fiscal pressures accumulated.

**Misconception 3: "MSCI removal happens slowly"**

Reality: Russia's removal from MSCI EM took eight days from announcement to effective date. MSCI has demonstrated a willingness to move extremely rapidly when market access is materially and immediately impaired. The criteria are market access (can foreign investors trade and repatriate?), liquidity, and custody/clearing safety. When all three fail simultaneously — as they did in Russia's case — MSCI acts within days. Investors should not assume months of warning.

**Misconception 4: "NDF markets always have enough liquidity to hedge"**

Reality: NDF liquidity is highly concentrated in a few major currencies. For smaller EM currencies experiencing the most acute access risk, the NDF market is often too thin or nonexistent. The currencies most in need of offshore hedging tools (IRR, MMK, EGP during the 2023 crisis) are precisely those for which the tools are least available. This is not a market failure to be fixed; it is the logical consequence of the fact that NDF market-making requires creditworthy counterparties willing to take two-way risk in a currency, and no bank wants to be a two-way market maker in a currency that might become worthless due to political discontinuity.

**Misconception 5: "Shutdowns only affect countries with weak institutions"**

Reality: India executed 84 internet shutdowns in 2022, the most of any country globally, predominantly in Jammu and Kashmir. India is a G20 economy with one of the world's largest stock markets, a deep INR NDF market, and MSCI EM membership at ~16% weight. The Kashmiri shutdowns are regional and have not materially impacted the national market, but they demonstrate that the tool is not reserved for authoritarian basket cases. Democracies use shutdowns too — the legal frameworks for them exist in India under the Indian Telegraph Act, and courts have been slow to restrict their use. If India were ever to deploy a shutdown that covered a major financial center, the INR NDF market would respond immediately.

## How It Shows Up in Real Markets

The practical signals an EM portfolio manager should monitor:

**Signal 1: Netblocks / AccessNow real-time shutdown alerts**

Netblocks publishes real-time internet disruption data. A confirmed multi-day shutdown announcement in a country with pre-existing capital controls should immediately trigger a review of that country's currency exposure. The expected move — 15-25% parallel-rate depreciation within 5 days — may not be fully accessible via traded instruments, but the signal should inform position sizing and option buying.

**Signal 2: NDF implied rate widening**

For countries with liquid NDF markets (INR, BRL, KRW, CNY), a widening of the offshore NDF rate versus the onshore spot is the most direct market signal of access risk repricing. An INR NDF premium of 0.5%+ above the onshore rate (versus a normal 0.1-0.2%) suggests institutional participants are pricing increased access risk. This can precede formal capital control announcements.

**Signal 3: CDS spread widening on sovereign debt**

Sovereign CDS spreads incorporate both credit risk and access risk. When Russia's CDS went from 150bps to 3000bps in the first week of the Ukraine invasion, it was partly reflecting credit risk (willingness to pay) and partly access risk (ability to pay, and ability of creditors to collect). Monitoring the ratio between a country's CDS spread and its peer group can signal early-stage access risk repricing.

**Signal 4: Roskomnadzor / regulatory announcements**

For Russia specifically, the pipeline of regulatory actions — blocking orders, DPI installation requirements, new ISP licensing rules — is the leading indicator for the next round of market-access restrictions. For Iran, the status of JCPOA negotiations and US sanctions policy is the primary driver. For Myanmar, the military council's periodic crackdowns correlate with shutdown waves. Each country has a specific set of leading indicators that precede the market move.

**Signal 5: MSCI market access review announcement**

MSCI publishes its annual market classification review results each June, with a consultation period from February. Countries placed on the "watch list" for potential downgrade appear in the February consultation document. Historically, there is a tradeable spread compression between the announcement of a review and the actual downgrade decision — passive vehicles begin reducing exposure in anticipation, creating temporary liquidity windows.

## How to Trade It — The Playbook

The core insight from the foregoing analysis is that internet shutdowns and capital controls create identifiable, bounded events with predictable market consequences. The playbook has four components.

**Component 1: Pre-position in NDF premium widening**

For countries with active NDF markets and elevated political risk (India-Kashmir tension escalation, China-Taiwan strait activity, Turkey election uncertainty), the NDF-onshore spread is the primary tradeable instrument. The trade structure:

- Buy the NDF at current spot (paying the current premium)
- Hold through the political event
- If the event materializes (shutdown, controls, sanctions announcement), the NDF rate will spike against spot as access risk reprices
- If the event does not materialize, roll the position or close at a small loss (the premium cost)

The cost of carry for this trade — the premium built into the NDF above spot — is the explicit price of the option. For major EM currencies in normal conditions, this might be 0.1-0.3% per quarter. That is the explicit cost of protection.

**Component 2: Buy put options on EM ETFs with country-specific exposure**

For MSCI EM-heavy plays on specific country risk, ETFs provide liquid proxy instruments. The iShares MSCI India ETF (INDA), iShares MSCI Brazil ETF (EWZ), and similar single-country ETFs have liquid options markets. Buying puts on these ETFs ahead of expected political stress events is a clean way to express the view without navigating NDF market mechanics.

The limitation: most country ETFs are broadly diversified, so the put options will only partially capture a pure currency-access-risk move (they will also capture equity market movements, which may move independently).

**Component 3: Short sovereign CDS on countries with active shutdown infrastructure**

For institutional investors with CDS access, sovereign CDS represents the cleanest expression of access risk for countries where the debt is held by foreigners. A country that shuts down its internet and imposes capital controls is also more likely to default on — or restructure — its external obligations. Buying protection via CDS is often cheaper before the event materializes than after.

**Component 4: Monitor the MSCI weight calendar**

If a country is placed on the MSCI review watch list, the forced-selling dynamic from passive vehicles creates a predictable market pattern:

- Announcement of watch list inclusion: immediate selling by active managers who anticipate the downgrade
- Consultation period (February-June): continued drift lower in the country's equity prices as uncertainty persists
- June classification decision: if downgrade confirmed, fresh forced selling from passive vehicles in the 5-30 days between announcement and effective date
- Effective date: passive vehicles complete rebalance, creating potential reversal opportunity if the downgrade was fully priced

This is not a unique insight — the pattern is well-known enough that there are dedicated hedge fund strategies around index-inclusion/exclusion events. But it is worth noting that the access-risk version of this trade (where the removal is driven by shutdown/sanctions/capital controls rather than a pure liquidity reclassification) tends to be more violent and faster-moving than the typical index event.

## Further Reading and Cross-links

For readers who want to go deeper on the individual components of this framework:

**On internet shutdowns and documentation:**
The Access Now KeepItOn coalition publishes annual reports tracking every documented shutdown by country, duration, stated justification, and estimated affected population. Netblocks provides real-time monitoring at the network measurement level. Both are essential primary sources.

**On NDF market mechanics:**
The BIS Triennial Survey is the definitive source for NDF market size and structure, updated every three years. The IMF publishes annual reports on capital account restrictions through its Annual Report on Exchange Arrangements and Exchange Restrictions (AREAER). For NDF settlement mechanics and documentation, ISDA's standard documentation for FX derivatives is the legal reference.

**On MSCI market access criteria:**
MSCI publishes detailed descriptions of its market accessibility criteria — including minimum requirements for foreign ownership limits, capital flow restrictions, and operational framework — in its annual Global Market Accessibility Review. Russia's 2022 removal decision document is a useful case study in how extreme access events trigger rapid reclassification.

**On Russia RuNet and sovereign internet laws:**
Access Now and Freedom House both publish detailed country-level reports on network sovereignty laws. Freedom House's Freedom on the Net report includes annual country scores and specific discussion of sovereign internet infrastructure.

**On Turkish lira capital controls and historical parallels:**
The Central Bank of the Republic of Turkey publishes historical exchange rate data and policy documents. For historical parallels — Argentina's corralito (2001-2002), Malaysia's ringgit peg (1998) — Barry Eichengreen's work on capital controls and currency crises provides the academic framework.

**Cross-links within this series:**
- Russia's SWIFT exclusion and the broader sanctions architecture are covered in the companion post on financial warfare and sanctions design.
- The Myanmar YSX halt and broader frontier markets fragility connects to the post on frontier market liquidity and the risks of thin-market investing.
- The NDF premium framework connects to the fixed income series post on emerging market bond risk premia and political risk decomposition.

The central lesson of this post — that market access is not binary but layered, and that the NDF market exists precisely because no other instrument survives all four layers of access denial — is both a technical insight and a geopolitical one. The countries most likely to deploy internet shutdowns are also the countries most likely to impose capital controls, and both behaviors tend to cluster around moments of political instability. Building a monitoring framework that watches for the leading indicators — network disruption signals, NDF spread widening, CDS spikes, MSCI review events — gives a risk-aware EM investor a genuine edge in a world where the next shutdown is more likely than the last.

The "political VPN premium" is real, it is measurable, and it is tradeable. The question is whether you have the infrastructure — analytical, operational, and hedging — to act on it before the market has fully repriced the access risk event that just switched on.
