---
title: "Sanctions law: OFAC, the SDN list, and how a designation bites"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "How a single name added to OFAC's SDN list cuts an entity off from the dollar — and why secondary sanctions and dollar-clearing chokepoints can take an asset to zero overnight."
tags: ["sanctions", "ofac", "geopolitics", "regulation", "compliance", "dollar-system", "secondary-sanctions", "sdn-list", "geopolitical-risk", "trading"]
category: "trading"
subcategory: "Law & Geopolitics"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — Financial sanctions are economic warfare run through the plumbing of the dollar; when OFAC adds a name to the SDN list, that entity is cut off from the US financial system, and via secondary sanctions and dollar-clearing chokepoints, usually from the global one too.
>
> - A **designation** is a binary switch: a US Treasury office adds a name to a list, and overnight every US person must freeze that entity's assets and stop dealing with it. No trial, no notice, no grace period.
> - **Secondary sanctions** globalize the prohibition by threatening *third parties* — a foreign bank that keeps dealing with the target risks losing its own access to dollar clearing, which is existential. That fear is what turns a US law into a worldwide blockade.
> - The **50 percent rule** means a designation reaches every entity an SDN owns 50 percent or more of, even unnamed ones — so your "clean" holding can be blocked by a parent you never screened.
> - The number to remember: in **2022, OFAC added a record ~2,549 new designations** in a single year, and froze about **\$300bn of Russia's ~\$643bn in reserves**. A designation is the fastest way an asset goes to zero for a US investor.

On a single Friday afternoon in February 2022, the US Treasury published a list that erased tens of billions of dollars of market value before the next London open. Names that had traded freely for years — Russian banks, energy firms, oligarch holding companies — became, in the span of a press release, untouchable. A US fund manager holding their bonds could not sell them, because there was no longer a legal buyer. A European bank that had cleared their payments for a decade stopped answering their calls. The assets did not "fall." They were switched off.

That switch is the subject of this post. It is operated by an office most investors have never heard of — the **Office of Foreign Assets Control**, or **OFAC**, a small unit inside the US Treasury — and it is the most powerful financial weapon any government has ever wielded, not because of armies or treaties, but because of an accident of history: the world runs on dollars, and dollars run through American banks. Control the chokepoint and you control who gets to participate in the global economy.

For an investor, sanctions are not a foreign-policy abstraction. They are a *tail risk priced into everything that touches a target* — a Russian energy bond, a Chinese chip firm, an Iranian oil cargo, a crypto exchange that processed the wrong wallet. Understanding how a designation works tells you how an asset can go to zero overnight, why compliance officers can freeze a trade you thought was clean, and how to size the risk that the rules of the game change against a position you hold. Let's build it from the ground up.

![How an OFAC designation transmits from a single SDN listing to a global cutoff and an asset repricing toward zero](/imgs/blogs/sanctions-law-ofac-the-sdn-list-and-secondary-sanctions-1.png)

## Foundations: OFAC, the SDN list, and the legal machinery

Before we can trade the risk, we have to understand the machine. Every term below is a load-bearing piece of how a designation moves from a government office to your brokerage statement.

### OFAC — the office that runs the list

The **Office of Foreign Assets Control (OFAC)** is a division of the US Department of the Treasury. Its job is to administer and enforce **economic and trade sanctions** — measures that restrict or prohibit US persons from dealing with specific countries, governments, entities, and individuals deemed a threat to US national security or foreign policy. Think of OFAC as the bouncer at the door of the US financial system: it maintains the guest list of who is barred, and it has the legal authority to penalize anyone who lets a barred party in.

OFAC is small — a few hundred staff — but its leverage is enormous, because it does not have to physically seize anything. It simply *publishes a name*, and the entire weight of US financial law falls on that name automatically.

The contrast with traditional statecraft is the whole point. A trade embargo enforced by a navy requires ships, ports, and the willingness to use force. A financial sanction requires a database update. The navy can blockade one harbor; the database update blocks the target from the system through which roughly **half of all global trade is invoiced and the large majority of cross-border bank claims are denominated**. This asymmetry — that a publishing decision can be more consequential than a fleet — is why sanctions have become the default tool of US economic statecraft over the past two decades. It is cheap to deploy, fast, scalable from one person to a whole country, and, critically, it does not require anyone else's permission. The US does not need the UN, NATO, or its allies to designate a name; it needs only the dollar's centrality, which it already has.

### The SDN list — the master blacklist

The central instrument is the **Specially Designated Nationals and Blocked Persons List**, universally called the **SDN list**. When OFAC "designates" a person or entity, it adds them to this list. The legal consequences of being on it are severe and immediate:

- All of the SDN's **property and interests in property** that are in the United States, or in the possession or control of any US person (anywhere in the world), are **blocked** — frozen in place. You cannot pay them, send them money, or move their assets.
- **US persons** — US citizens and permanent residents wherever located, anyone physically in the US, and any entity organized under US law including foreign branches — are **prohibited from any transaction or dealing** with the SDN. Not just payments: any dealing at all, including providing services or facilitating someone else's transaction.

A "blocking" is not a fine or a temporary hold. It is an indefinite freeze: the asset sits, untouchable, until OFAC removes the designation (delisting) or issues a license to release it. There is no judge, no trial before the listing, and no advance notice — the SDN typically learns they are designated when their accounts stop working.

The phrase "property and interests in property" is broader than it sounds, and the breadth matters for investors. It covers not just bank balances and securities but contracts, receivables, intellectual property, leases, escrow, and — importantly — *anything in which the SDN has an interest*, which is why a partially-owned asset can be caught (more on that under the 50 percent rule). "Blocked" also does not mean "confiscated": the US does not (usually) take title to the asset; it freezes it. The SDN technically still owns the frozen property, but can do nothing with it, and neither can anyone else. The asset is in legal limbo. For a US custodian holding a designated bond, this means the bond stays on the books, marked at recoverable value, generating no cash flow, with no permitted way to dispose of it absent a license.

A second subtlety: the SDN list is *one* of many lists OFAC and other agencies maintain, and a compliance system screens against all of them. There is the **SSI list** (sectoral, discussed below), the **Non-SDN Chinese Military-Industrial Complex Companies list** (which restricts US investment in named firms without fully blocking them), the **Foreign Sanctions Evaders list**, and others, plus the Commerce Department's **Entity List** (export controls — a separate regime that restricts what technology can be shipped to a named party). For trading purposes, the SDN list is the most severe, but a name can appear on a "lighter" list first — an early warning that a full blocking designation may follow.

### The legal authorities — where the power comes from

OFAC does not invent this power; it is delegated by statute and executive order. The two pillars an investor must know:

- **IEEPA — the International Emergency Economic Powers Act (1977).** This is the workhorse. It lets the President declare a *national emergency* with respect to an "unusual and extraordinary threat" originating largely outside the US, and then regulate or prohibit a vast range of economic transactions in response. Most modern sanctions programs — Russia, Iran, Venezuela, terrorism, narcotics — rest on an IEEPA-based executive order. The emergency is renewed annually; some have run for decades.
- **Country and thematic programs.** Beyond IEEPA, specific statutes target specific situations: the Trading with the Enemy Act (older, now mostly Cuba), the Global Magnitsky Act (human-rights and corruption designations), CAATSA (the 2017 act that codified and toughened Russia/Iran/North Korea sanctions), and many program-specific laws. Each authorizes designations under defined criteria.

The practical point: sanctions are **administrative law executed at the speed of a press release**, backed by statutes that grant the executive branch sweeping discretion. There is no legislative debate before a name is added. A designation can be announced before markets open and binding by the time they do.

There is one more legal wrinkle with direct market consequences. Because designations rest on an executive emergency declaration rather than a statute passed by Congress, they can be *reversed* by the same executive almost as quickly as they were imposed. A new administration, or a diplomatic deal, can lift sanctions with another stroke of the pen — which is exactly what makes the *delisting* trade (betting that a designation will be removed) a real, if dangerous, strategy. Sanctions are not permanent fixtures of the law; they are policy instruments, and policy changes. The Iran nuclear deal (the JCPOA) lifted a swathe of sanctions in 2016, only for many to be reimposed in 2018 — a whipsaw that repriced Iranian-exposed assets twice in two years. The lesson for the investor: a designation's *political durability* is as important as its legal mechanics, because the political will behind it is what determines how long the asset stays at zero.

### Primary vs secondary sanctions — who is bound

This distinction is the single most important thing in the whole topic, so read it twice.

- **Primary sanctions** bind **US persons**. If you are a US citizen, a US company, or a foreign branch of a US bank, primary sanctions prohibit *you* from dealing with the target. A French company with no US ties is, under primary sanctions, technically free to keep trading with the target.
- **Secondary sanctions** are aimed at **third parties — non-US persons.** They do not directly prohibit the foreign company; instead they threaten *consequences* if that foreign company deals with the target. The classic consequence: OFAC can designate the foreign company *itself* (put it on the SDN list), or bar it from the US financial system, or sanction its access to dollar clearing. In effect, the US says to a foreign bank: "We can't order you to stop dealing with Iran, but if you do, you can't deal with us — and we are the dollar."

Primary sanctions are a domestic prohibition. Secondary sanctions are how the US *exports* its sanctions to the rest of the world without an international treaty. We will see exactly how this transmits in a moment.

A historical note clarifies why this distinction is so loaded. For most of the 20th century, US sanctions were primary-only — they bound Americans, and the rest of the world could do as it pleased. Secondary sanctions, in their modern aggressive form, emerged largely with the Iran programs of the 2000s and 2010s, when the US decided that primary sanctions alone were not isolating Iran enough. By threatening foreign banks' dollar access, the US effectively conscripted the world's financial system into enforcing American foreign policy. Allies bristled — the EU even passed a "blocking statute" that purports to forbid European firms from complying with certain US secondary sanctions — but the blocking statute is largely toothless in practice, because no European bank will defy OFAC and risk its dollar business to obey a Brussels regulation it can quietly ignore. The dollar wins. That is the uncomfortable reality every non-US investor and corporate operates under: when European law and US secondary sanctions conflict, the dollar-clearing chain decides, and the dollar-clearing chain is American.

### The 50 percent rule — the silent reach

OFAC's **50 Percent Rule** states: any entity owned, directly or indirectly, **50 percent or more** in the aggregate by one or more blocked persons is *itself* considered blocked — **even if it is not named on the SDN list.** The reasoning is obvious: otherwise an SDN could simply route everything through an unlisted subsidiary. The trap for investors is equally obvious: you can screen a company against the SDN list, get a clean hit, and still be dealing with a blocked entity because its *parent* (or a coalition of parents) crosses the 50 percent threshold.

The rule is aggregated (two 30-percent SDN owners = 60 percent = blocked) and it flows downward through the ownership chain. A 51-percent-owned subsidiary of a 51-percent-owned subsidiary is blocked. A 49-percent stake, by contrast, is not automatically blocked — though OFAC warns you to exercise caution near the line, and a non-blocked entity can still be designated separately on its own conduct.

Two refinements that catch people out. First, the rule is about *ownership*, not *control* — but OFAC's guidance also warns that a blocked person who *controls* an entity (even below 50 percent ownership) creates significant risk, and OFAC may designate that entity for being controlled by an SDN. So 49 percent is not a safe harbor; it is merely outside the automatic rule. Second, the aggregation is across *all* blocked owners combined: if three different SDNs each hold 20 percent of a company, that company is 60 percent blocked-owned and is itself blocked, even though no single SDN owns a majority. For an investor screening a complex emerging-market conglomerate with opaque ownership, this is genuinely hard work — you may need beneficial-ownership data several layers deep, and that data is often the exact thing the structure was built to obscure. The practical defense is to treat *ownership opacity itself* as a risk factor: a holding whose ultimate beneficial owners you cannot fully map is a holding whose 50-percent-rule exposure you cannot rule out.

### Blocking sanctions vs sectoral sanctions

Not every sanction is a total blackout. Two flavors:

- **Blocking (full) sanctions** are the SDN-list treatment described above: total freeze, no dealings. This is the nuclear option.
- **Sectoral sanctions** are surgical. Rather than blocking an entity entirely, they prohibit *specific* types of transactions — for example, providing new long-term financing or equity to a named Russian bank, while still allowing short-term trade payments. The 2014 US sanctions on Russia after Crimea used **Sectoral Sanctions Identifications (the SSI list)** to choke off capital markets access without freezing the firms outright. Sectoral sanctions are a dimmer switch; blocking sanctions are an off switch.

### The dollar-clearing chokepoint — why everything routes through US banks

Here is the geopolitical engine under the hood. Most cross-border trade and finance is denominated in US dollars — oil, commodities, a huge share of international loans and trade invoices. But a dollar is, ultimately, a liability of the US banking system. When a bank in Brazil pays a bank in Vietnam in dollars, the actual settlement does not happen in Brazil or Vietnam — it happens on the books of a **US correspondent bank** (and through US clearing systems like **CHIPS** and **Fedwire**). The foreign banks hold dollar accounts with US banks; moving dollars means moving balances between those accounts inside the US.

This is **correspondent banking**, and it is the chokepoint. Because the dollar leg of nearly every international payment touches a US bank, OFAC can screen and block payments at that single point. Cut a target off from US correspondent banking and you have cut them off from dollars — and from most of global trade.

![Two foreign banks settling in dollars must route through a US correspondent bank, the single point OFAC can switch off](/imgs/blogs/sanctions-law-ofac-the-sdn-list-and-secondary-sanctions-2.png)

This is why the US dollar's reserve status is not just an "exorbitant privilege" of cheap borrowing — it is a *weapon*. We unpack the privilege side in [the petrodollar and dollar dominance](/blog/trading/finance/petrodollar-and-dollar-dominance); here the relevant point is that dominance creates a chokepoint, and the chokepoint is what makes sanctions bite globally.

It is worth being concrete about the plumbing, because the shorthand "the dollar" hides the real mechanism. There is no central pool of dollars in the sky. Every dollar is a balance somewhere in the US banking system. A bank in Lagos that wants to "hold dollars" actually holds a balance in an account at, say, a US bank in New York (its **correspondent**), or in an account at a larger foreign bank that *itself* has a New York correspondent. When the Lagos bank pays a supplier in Hanoi, the instruction travels through this chain of correspondents until two of them — both with accounts inside the US system — adjust their balances against each other on US soil, often via **CHIPS** (the Clearing House Interbank Payments System, which settles the large-value dollar payments) or **Fedwire** (the Federal Reserve's own settlement rail). The messaging that coordinates all this often rides on **SWIFT**, a Belgian-based interbank messaging network — but SWIFT only carries the *instruction*; the actual *settlement* of dollars happens in the US. That is why being cut off from SWIFT hurts, but being cut off from dollar correspondent banking is fatal: you can find another way to send a message, but you cannot find another place to settle a dollar. OFAC sits astride that settlement point, and that is the whole game.

### OFAC licenses — the exceptions

Sanctions are not always absolute. OFAC can authorize otherwise-prohibited transactions through **licenses**:

- A **general license** is a published, blanket authorization that lets *anyone* meeting stated conditions do a specified thing — for example, a wind-down license giving firms 30 or 90 days to exit existing contracts with a newly designated entity, or a humanitarian license permitting food and medicine.
- A **specific license** is a one-off, applied-for authorization granted to a particular party for a particular transaction.

Licenses matter enormously to traders, because a general wind-down license is often what determines whether you can *exit* a position in the days after a designation or whether you are simply frozen. The presence, scope, and expiry of a wind-down license is one of the first things to read after any major listing.

A wind-down license is best understood as a *legally sanctioned escape hatch with a closing door*. When OFAC designated major Russian entities in 2022, it simultaneously issued general licenses giving counterparties a window — often 30 to 90 days — to settle and exit existing positions and contracts. Inside that window, a US holder could (in principle) sell a designated bond to a non-US buyer in a legally compliant wind-down; after it expired, the position was frozen with no path out. The market consequence is that the *terms of the license*, not the designation alone, set the floor price during the wind-down: holders rush to exit before the door closes, which is exactly why prices on freshly designated paper crater within the license window rather than gradually. If you hold a name that gets designated, the wind-down license is the single document that determines whether your loss is "large and realized" or "total and frozen." Read it before you read anything else.

### The penalty regime — why compliance is not optional

Violating sanctions carries both **civil** and **criminal** liability. Civil penalties under IEEPA can reach the greater of roughly **\$300,000+ per violation** (the cap is inflation-adjusted and rises over time) **or twice the value of the underlying transaction** — and a single course of dealing can be carved into many "violations." Criminal penalties for willful violations run to **\$1 million per count and up to 20 years in prison**. Crucially, the standard for civil liability is effectively **strict** — you do not have to *intend* to violate sanctions; processing the transaction is enough. That strict-liability posture is the reason banks behave the way they do, which we turn to next.

The per-violation multiplication is what turns these numbers from large to ruinous. Consider a bank that processed 1,000 prohibited dollar payments over several years. At twice the transaction value per payment, even modest payments aggregate into enormous totals — which is precisely how settlements reach the billions. OFAC also publishes "enforcement guidelines" that reward **voluntary self-disclosure** and a strong compliance program with large penalty reductions, and punish willful or reckless conduct with penalty increases. The asymmetry is deliberate: a firm that catches its own violation, reports it, and fixes the gap might settle for a fraction of the base penalty; a firm that hid the conduct and was caught faces the full multiplier plus criminal referral. This guideline structure is itself a behavioral lever — it pushes every regulated institution to invest heavily in detection and self-policing, because the cheapest path through a violation is to find and confess it first. For the investor, the takeaway is that the *threat* of these penalties is doing the work long before any penalty is actually levied: the entire global banking system is pre-configured to over-react to sanctions risk because the downside of under-reacting is existential.

## How a designation transmits: from a press release to a global blockade

We now have the parts. Let's trace exactly how a name on a list becomes a worldwide cutoff, in the order it actually happens.

### Step 1: US persons freeze instantly (primary sanctions)

The moment OFAC publishes the designation, every US person is, by operation of law, prohibited from dealing with the target and must block any of the target's property in their control. This is automatic and immediate. Your US broker's compliance system screens holdings against the SDN list (updated daily); a hit triggers an instant freeze. You cannot sell the security, transfer it, or receive a dividend on it. For a US investor, the asset is not "down 90 percent" — it is *inaccessible at any price*.

### Step 2: Secondary sanctions threaten the rest of the world

Primary sanctions alone would leave a target free to do business with everyone outside the US. Secondary sanctions close that door. By threatening to designate *foreign* parties who continue dealing with the target — and to cut those foreign parties off from dollar clearing — OFAC gives every non-US bank and corporate a choice: keep the (small) business with the target, or keep access to the (enormous) US dollar system. For any institution that wants to function in global finance, that is not a real choice.

![Primary sanctions bind only US persons while secondary sanctions reach every foreign party that deals with the target](/imgs/blogs/sanctions-law-ofac-the-sdn-list-and-secondary-sanctions-3.png)

### Step 3: The chilling effect and over-compliance

Here is the second-order effect that makes sanctions far more powerful than their literal text. Because penalties are effectively strict-liability and enormous, and because the cost of one mistake (losing dollar-clearing access) is existential, banks do not calibrate to the *letter* of the rules. They **over-comply**. Faced with even a *whiff* of sanctions exposure, a compliance department will simply refuse the business, exit the relationship, or — at the extreme — **de-risk an entire country**: closing correspondent relationships with all banks in a jurisdiction because the cost of vetting each one exceeds the revenue.

This is why sanctions have ripple effects far wider than their targets. A small Pacific nation can lose its only dollar correspondent because a global bank decides the whole region is "high risk." The targets pay first; the bystanders pay next. For an investor, this means a designation in a sector or a country can taint *adjacent* assets that were never named — a contagion of compliance.

The economics of over-compliance are worth spelling out because they explain a market behavior that otherwise looks irrational. A compliance officer at a global bank faces a brutally asymmetric payoff. If she correctly clears a borderline-but-legal payment, the bank earns a tiny fee and she earns nothing personally. If she wrongly clears a payment that turns out to violate sanctions, the bank can face a multi-billion-dollar penalty, the headline is the bank's name in a Treasury press release, and *she* may lose her job or face personal liability. Given that payoff matrix, the rational officer refuses anything ambiguous. Multiply this across every compliance officer at every bank in the world and you get a system that systematically *under-serves* legal-but-risky clients — entire categories of legitimate business (remittances to certain countries, correspondent banking for small jurisdictions, accounts for non-profits operating in conflict zones) get cut off not because they are illegal but because they are *not worth the tail risk*. Economists call this the **de-risking** problem, and it is a documented, measurable shrinkage of correspondent banking relationships in the riskier corners of the world. For an investor, the signal is that sanctions risk prices into assets *before and beyond* any actual designation: a bank, a payments firm, or an economy heavily dependent on these flows carries a structural discount for the de-risking tail.

### Step 4: The 50 percent rule reaches the unnamed

In parallel, the designation silently propagates down ownership chains. Any subsidiary that is 50 percent or more owned by the SDN is blocked, even though OFAC never typed its name. A fund holding what looked like a clean operating company can discover that a newly designated parent has just blocked the subsidiary — and therefore the fund's position in it.

![A designation reaches every entity an SDN owns 50 percent or more of, including unnamed subsidiaries down the chain](/imgs/blogs/sanctions-law-ofac-the-sdn-list-and-secondary-sanctions-4.png)

### Step 5: The asset reprices — toward zero, fast

Put these together and the market impact is not a slide; it is a step function. For US holders the asset is frozen (unsellable). For foreign holders it is radioactive (no one wants the compliance risk of buying it). Index providers delete it. Market-makers withdraw quotes. The bid simply disappears. A bond that was 95 cents on the dollar on Thursday can be marked at a few cents — or untradeable entirely — by the following week, not because the issuer's fundamentals changed, but because the *legal right to own and trade it* was revoked.

This is the feature that most distinguishes sanctions risk from ordinary market risk, and it is the one investors most consistently underestimate. Ordinary risk is *continuous*: bad news arrives, the price falls, and you can sell at the new (lower) price — painful, but tradeable. Sanctions risk is *discontinuous and right-removing*: the news doesn't lower the price, it abolishes the market. There is a profound difference between "this bond is now worth 10 cents" (you can sell at 10) and "you are legally prohibited from selling this bond" (you can sell at nothing). A liquidity crisis dries up the bid temporarily; a designation removes the *legal capacity* to be a buyer or seller, which no amount of liquidity returning can fix until the designation is lifted. This is why you cannot "stop-loss" your way out of sanctions risk. A stop-loss assumes there is a price at which someone will take the other side; a designation removes the other side entirely. The only defenses are *not holding the tail* in size, or *exiting inside the wind-down window* if one exists. Once the window closes, the position is not a loss you can crystallize — it is an asset you can no longer touch.

![From a designation on day zero to a frozen, bid-less, marked-to-zero position within a week](/imgs/blogs/sanctions-law-ofac-the-sdn-list-and-secondary-sanctions-7.png)

#### Worked example: a designated company's equity going to ~zero for a US investor

Suppose a US fund holds **\$5,000,000** of a foreign-listed company's stock, bought at \$50 a share — **100,000 shares**. On Friday after the close, OFAC designates the company and adds it to the SDN list. What happens to that position?

On Monday, the fund's prime broker, screening holdings against the updated SDN list, blocks the position. The fund cannot sell at \$50, cannot sell at \$5, cannot sell at all — there is no licensed US buyer, and the broker is legally barred from executing the trade. If a 30-day general wind-down license were issued, the fund *might* unwind to a non-US buyer at a distressed price; absent one, the position is frozen indefinitely.

For mark-to-market purposes, the fund must write the position down to its recoverable value. With no legal market, that is approximately **\$0**:

```
Pre-designation value:  100,000 shares x $50  = $5,000,000
Recoverable value (no legal buyer, frozen)    ~ $0
Loss realized on the fund's books             = $5,000,000  (-100%)
```

The lesson: a designation is not a price move you can trade around — it revokes the *right to transact*, and a security you cannot legally sell is worth approximately nothing to you no matter what it "should" be worth.

#### Worked example: the 50 percent rule reaching a subsidiary

A US asset manager holds **\$8,000,000** of bonds issued by "CleanCo," an operating company that is **not** on the SDN list. CleanCo screens clean. But CleanCo is **60 percent owned** by "ParentHoldings," and on Tuesday OFAC designates ParentHoldings.

Under the 50 Percent Rule, because ParentHoldings (a blocked person) owns ≥ 50 percent of CleanCo, **CleanCo is automatically blocked too** — without ever being named. The manager's \$8,000,000 in CleanCo bonds is now frozen.

```
CleanCo ownership:  ParentHoldings 60%  (>= 50% threshold)  -> CleanCo BLOCKED
Position frozen:    $8,000,000 of CleanCo bonds, immediately
Screening against the SDN list alone:  MISSED IT  (CleanCo never listed)
```

Now flip one number. Had ParentHoldings owned only **49 percent**, CleanCo would *not* be automatically blocked — the same \$8,000,000 stays tradeable (subject to ongoing diligence). One percentage point of ownership is the difference between a frozen position and a free one. The lesson: screen the *ownership tree*, not just the name — the 50 percent rule is where "clean" holdings get caught.

## The compliance cost — what sanctions do to everyone who isn't the target

The target is cut off. But the *machinery* of staying compliant imposes a permanent tax on every bank and corporate that touches international flows.

Banks must screen every customer and every payment against the SDN list (and dozens of other lists), maintain ownership-tree data to apply the 50 percent rule, file blocking reports with OFAC, and staff large compliance and "financial crime" departments. The largest global banks spend **billions of dollars a year** on sanctions and anti-money-laundering compliance combined, and employ tens of thousands of staff in those functions. For a corporate treasurer, sanctions screening adds latency and refusal risk to ordinary cross-border payments.

And the penalties for getting it wrong are not theoretical. BNP Paribas paid **\$8.9 billion** in 2014 for systematically processing dollar transactions for sanctioned Sudan, Iran, and Cuba. Standard Chartered, HSBC, Commerzbank, and others have paid hundreds of millions to billions each. These settlements are why compliance budgets are what they are — and why, as we saw, banks over-comply rather than risk being the next headline. We dig into the enforcement and evasion side in [sanctions evasion, enforcement, and compliance risk](/blog/trading/law-and-geopolitics/sanctions-evasion-enforcement-and-compliance-risk).

For the equity investor, this compliance cost is not just a story about banks — it is a *line item* and a *risk factor* in any firm that moves money across borders. A global bank's compliance and financial-crime headcount can run into the tens of thousands; the associated technology, data subscriptions, and audit costs are a permanent drag on return on equity that a domestic-only competitor does not carry. When you value a multinational bank, payments processor, money-transfer operator, or crypto exchange, the sanctions-compliance burden belongs in your cost structure — and the *tail* of a sanctions settlement belongs in your risk assessment, because a single multi-billion-dollar penalty can wipe out a year or more of earnings and trigger a regulatory consent order that constrains the business for years afterward. The cleanest way to see this: a sanctions settlement is, for a bank, simultaneously a one-time charge (the fine), an ongoing cost increase (the remediation and monitoring the settlement mandates), and a reputational/franchise hit (clients and correspondents reassess the relationship). All three reprice the equity, and none of them show up in a simple price-to-book screen until after the headline.

#### Worked example: a foreign bank's secondary-sanctions choice

A mid-sized foreign bank earns roughly **\$4 million a year** in fees clearing trade payments for a newly sanctioned client. Tempting to keep — until you size the other side of the ledger. The bank runs a **\$60 billion** annual dollar-clearing business through its US correspondent, generating, say, **\$120 million a year** in associated revenue and underpinning its entire international franchise. If it keeps the sanctioned client and OFAC notices, the bank risks being cut off from dollar clearing entirely.

```
Keep the client:   +$4,000,000 / yr in target fees
Downside if caught: lose ~$120,000,000 / yr dollar-clearing revenue
                    + an OFAC fine (BNP precedent: $8.9bn)
                    + the franchise itself

Expected value of keeping the client (even at 10% catch probability):
  +$4,000,000  -  0.10 x ($120,000,000 + fine)  <<  0
```

![A foreign bank compares small target fees against losing its entire dollar franchise, so it over-complies](/imgs/blogs/sanctions-law-ofac-the-sdn-list-and-secondary-sanctions-8.png)

The math is so lopsided that the rational bank doesn't just drop the client — it drops anything that *looks like* the client. The lesson: secondary sanctions work not by being enforced often, but by making the downside so catastrophic that no profit-seeking institution will go near the line.

![New OFAC designations per year rose sharply, with a record 2,549 added in 2022 around Russia's invasion](/imgs/blogs/sanctions-law-ofac-the-sdn-list-and-secondary-sanctions-5.png)

The chart above shows the scale of the tool. OFAC designations have trended up for a decade, and **2022 was a record year — roughly 2,549 new listings** — driven overwhelmingly by the Russia program. Sanctions are not a rarely used emergency lever; they are an increasingly routine instrument of statecraft, which means the designation tail risk is *rising* across emerging-market and geopolitically exposed assets.

## Sanctions on a country vs a person vs a sector

The same legal machinery scales from one person to an entire economy. Three modes:

- **A person or entity.** A single oligarch, a single bank, a single company is added to the SDN list. The blast radius is the target plus its ≥ 50 percent-owned subsidiaries plus its direct counterparties' willingness to deal.
- **A sector.** Sectoral sanctions target a *category* of activity across many firms — for example, new financing for Russian energy and defense companies, or US-listing and capital-raising bans on Chinese military-linked firms. Surgical, but broad: they reprice an entire industry's cost of capital.
- **A country.** Comprehensive programs (historically Iran, North Korea, Cuba, Syria; parts of the Russia and Venezuela programs) effectively wall off a whole economy. **Iran's oil exports** are the textbook case: US secondary sanctions on buyers of Iranian crude pushed most Western refiners out entirely, forcing Iran to sell to a shrinking pool of buyers at steep discounts through opaque channels. **Venezuela's oil sector** sanctions similarly stranded its barrels and collapsed its export revenue. **Russia's 2022 program** combined all three — blocking individual banks and people, sectoral capital-markets restrictions, and near-comprehensive measures plus an oil price cap.

The investing relevance: the *mode* tells you the blast radius. A single-name designation is an idiosyncratic, hedgeable risk. A sector or country program is a regime change for an entire asset class — and it tends to reroute physical flows (oil, gas, metals) along new, discounted channels.

The sectoral mode deserves a closer look because it is where the most *tradeable* (rather than simply catastrophic) opportunities live. Unlike a full blocking designation, sectoral sanctions leave the targeted firms trading — they just raise the firms' cost of capital and shrink their addressable market. The 2014 SSI program on Russia, for example, didn't freeze Sberbank or Rosneft; it barred US persons from providing them with new long-term debt or equity. The effect was a *re-rating*: those firms' cost of capital jumped because half the world's capital pool was suddenly off-limits to them, their bond spreads widened, and their equity discounted the lost growth from capital starvation. That is a *gradient*, not a cliff, and a gradient can be analyzed, sized, and traded the way you'd trade any cost-of-capital shock — short the squeezed names, demand a higher yield to hold their paper, or play the relative value between a sanctioned firm and an unsanctioned peer. Sectoral sanctions are the part of the sanctions toolkit that behaves most like ordinary regulation: they change the rules of the game gradually rather than ending it instantly.

#### Worked example: a binary designation event and expected value

You're considering a high-yield bond of an emerging-market energy firm that sits in a geopolitically tense region. It trades at **90 cents on the dollar** and offers a fat yield. Your fundamental work says that *absent sanctions* it's worth **95 cents**. But there is a real chance the firm gets designated over your one-year horizon, in which case it goes to roughly **5 cents** for you (frozen, near-zero recovery). How do you think about the price?

Treat the designation as a **binary event** with probability *p*. The fair value is the probability-weighted average of the two outcomes:

```
No designation (prob 1 - p):   worth 95 cents
Designation     (prob p):       worth  5 cents

Fair value = (1 - p) x 95  +  p x 5

If p = 5%:   0.95 x 95 + 0.05 x 5  = 90.25 + 0.25 = 90.5 cents  -> bond at 90 is roughly fair
If p = 15%:  0.85 x 95 + 0.15 x 5  = 80.75 + 0.75 = 81.5 cents  -> bond at 90 is RICH by ~8.5 cents
If p = 30%:  0.70 x 95 + 0.30 x 5  = 66.5  + 1.5  = 68 cents    -> bond at 90 is wildly overpriced
```

The market price of 90 is only justified if the designation probability is around 5 percent. The moment your read of the escalation ladder pushes *p* toward 15 or 30 percent — a new round of sectoral sanctions, a major counterparty designated, rhetoric turning to action — the bond is sharply overpriced and you should be a seller, not a yield-chaser. The lesson: a fat yield is not free money when the tail is a binary "right-to-own" event; you must back out the *implied designation probability* from the price and decide whether the real-world probability is higher.

#### Worked example: sanctioned oil sold at a discount and the buyer's pickup

When a major oil exporter is sanctioned, its crude doesn't vanish — it gets **rerouted to non-aligned buyers at a discount** to compensate them for the sanctions risk, shipping complications, and payment friction. Suppose the global benchmark (Brent) is **\$85/bbl**, and the sanctioned grade trades at a **\$25/bbl discount**, i.e. **\$60/bbl**, to clear.

A refiner in a non-sanctioning country that buys **500,000 barrels** captures the discount directly:

```
Brent benchmark:        $85 / bbl
Sanctioned grade:       $60 / bbl   (a $25 / bbl discount)
Cargo size:             500,000 bbl

Discount captured:  500,000 x $25  = $12,500,000 per cargo
```

The buyer pockets **\$12.5 million** per cargo for taking on the sanctions and logistics risk — and as long as that buyer is outside the reach of secondary sanctions (or judges the enforcement risk acceptable), the trade clears. This is precisely the dynamic behind the discounted flows that followed the Russia oil sanctions, which we examine in [the 2022 Russia sanctions and the weaponization of finance](/blog/trading/law-and-geopolitics/the-2022-russia-sanctions-and-the-weaponization-of-finance). The lesson: sanctions rarely stop a commodity from flowing — they tax it, splitting the price into a "compliant" tier and a discounted "sanctioned" tier, and someone always shows up to capture the spread.

## Common misconceptions

Three beliefs that cost investors money, each corrected with numbers.

**Misconception 1: "Sanctions only bind US firms, so a non-US investor is safe."** This is the most expensive misunderstanding in the topic. *Primary* sanctions bind US persons — but *secondary* sanctions deliberately reach the rest of the world by threatening dollar-clearing access. When BNP Paribas, a **French** bank, paid **\$8.9 billion** in 2014, it was for processing dollar payments for sanctioned countries — conduct that was legal under French law but exposed BNP to the US because the transactions cleared through New York. A European fund holding a sanctioned bond faces the same reality: its custodian and clearing banks will freeze the position regardless of where the fund is domiciled, because *they* touch the dollar system. There is no "non-US safe harbor" for anything that clears in dollars.

**Misconception 2: "You can trade around sanctions easily — just route through a third party."** Evasion is possible but expensive, and the costs are exactly what makes sanctions work. The 50 percent rule blocks the obvious workaround (an unnamed subsidiary). The chokepoint blocks the dollar workaround. And enforcement is severe and retrospective: OFAC settled with numerous firms for transactions years after the fact, and the largest penalties run into the **billions**. The discounted oil example shows the *real* cost of working around sanctions — the sanctioned exporter gave up roughly **\$25 of every \$85 barrel** (a ~30 percent haircut) precisely because routing around the chokepoint is not free. "Easily" is wrong; "at a large, persistent cost" is right.

**Misconception 3: "Sanctions always work — the target collapses."** Sanctions are powerful at *cutting off* a target from the dollar system, but "working" in the sense of changing a government's behavior is a much weaker claim. Iran, North Korea, Cuba, and Venezuela have endured comprehensive sanctions for years or decades without regime change or policy reversal; targets adapt by rerouting trade, building alternative payment channels, and shifting toward non-dollar partners. The investing takeaway is precise: sanctions *reliably* reprice the targeted assets (toward zero for US holders) and *reliably* impose costs, but they do *not* reliably achieve their stated geopolitical goal. Don't confuse "the bond went to zero" (almost certain) with "the regime will fold" (often wrong). The persistence of sanctioned regimes is also what fuels the de-dollarization debate — covered in the Russia case study.

**Misconception 4: "Once you're cut off, you can never come back."** Designations are removable, and the *delisting* is itself a market event. OFAC delists names regularly — sometimes because the underlying conduct stopped, sometimes as part of a negotiated deal, sometimes because a court or administrative review forced it. When a major name is delisted, the previously-frozen asset can reprice violently *upward* as the legal right to trade returns. The Iran sanctions relief under the 2016 nuclear deal repriced a swathe of Iranian-exposed assets higher; the 2018 reimposition repriced them back down. The point for an investor is symmetry: if a designation can take an asset from 90 to 5 overnight, a delisting can take it from 5 back toward fundamentals just as fast. This is what makes the *delisting trade* — buying deeply distressed sanctioned paper on a bet that sanctions will be lifted — a real strategy, though a brutally binary and politically-driven one. It is the mirror image of the designation tail, and it is priced by the same expected-value math: the distressed price embeds an implied probability of relief, and your edge is having a better read on the politics than the market.

## How it shows up in real markets

Three patterns recur. Recognize them and you can position before the crowd fully prices them.

**Pattern 1 — the SDN-designation collapse.** The cleanest example is the Russia program. When OFAC and allied authorities designated major Russian banks and the central bank in February–March 2022, Russian sovereign and corporate dollar bonds were marked down to single-digit cents within days, index providers (JPMorgan, MSCI, FTSE Russell) removed Russian securities from their benchmarks, and Western holders were left frozen. The price didn't reflect a change in Russia's ability to pay — it reflected the revoked *right* to hold and trade the paper. The same step-function collapse hits any single-name designation: the asset goes from a price to a freeze.

**Pattern 2 — the reserve freeze.** The most dramatic 2022 action was immobilizing roughly **\$300 billion of Russia's ~\$643 billion in foreign-exchange reserves** held in G7 jurisdictions. This was unprecedented: it showed that even a *sovereign's* dollar (and euro) reserves are only as safe as the issuer's willingness to let you use them. The chart below shows the split. The strategic consequence — every central bank in the world reassessing how much of its reserves to hold in dollars and euros — is the seed of the de-dollarization conversation we pick up elsewhere.

![About half of Russia's roughly 643 billion dollar reserves were frozen by G7 sanctions in February 2022](/imgs/blogs/sanctions-law-ofac-the-sdn-list-and-secondary-sanctions-6.png)

**Pattern 3 — the de-risking exit and the oil reroute.** Less visible but pervasive: after a major program, global banks quietly exit *adjacent* relationships, and physical commodity flows reroute to discounted channels. Western banks pulled back from financing trade across whole regions perceived as high-risk; Russian and Iranian crude found buyers in Asia at steep discounts to Brent. For a trader, the signal is the *discount* and the *flow shift* — the sanctioned commodity doesn't disappear from the global balance, it changes hands at a worse price, which is itself a tradeable spread for those legally able to capture it.

**Pattern 4 — the second-order rally.** Sanctions are a geopolitical shock, and geopolitical shocks lift a recognizable basket. Defense and aerospace equities rally as escalation implies higher military budgets — the iShares US Aerospace & Defense index rose roughly 38 percent from the eve of the 2022 invasion through end-2024, even as broad emerging-market and Russia-exposed assets collapsed. Energy producers outside the sanctioned region rally as supply tightens and prices rise. Safe havens — gold, the dollar, Treasuries — bid as capital seeks shelter. The Geopolitical Risk index spiked in February 2022 the way it spiked after 9/11 and the Iraq invasion, and the cross-asset response rhymed each time. For a portfolio, the lesson is that a sanctions event is rarely a single-name story; it is a *regime signal* that reprices an entire basket of geopolitically-sensitive assets in a correlated move, and the second-order winners (defense, energy, havens) are often the more liquid and tradeable expression of the shock than the frozen targets themselves.

## How to trade it: the sanctions-risk playbook

Sanctions are a tail risk that prices into everything touching a potential target. Here is how a practitioner reads and positions for it.

**1. Screen your holdings — name *and* ownership tree.** Run every position against the current SDN list (it updates daily), but don't stop at the name. Apply the **50 percent rule**: map the ownership of each holding up to its ultimate parents, and flag anything where blocked or potentially-blockable parents aggregate near or above 50 percent. The CleanCo example is the trap — clean name, blocked parent. For emerging-market and geopolitically exposed names, treat the ownership tree as a first-class risk input, not a compliance afterthought. Practically, this means subscribing to a screening service that maintains beneficial-ownership data (the major compliance-data vendors do), re-running the screen whenever the SDN list updates rather than only at purchase, and flagging any holding whose ownership you *cannot* fully resolve as a latent 50-percent-rule risk rather than assuming it is clean. Opacity is not safety; it is unmeasured exposure.

**2. Map secondary-sanctions exposure.** For each holding, ask: *who has to keep dealing with this for my thesis to hold, and are they exposed to secondary sanctions?* A company that depends on dollar financing, on Western banks clearing its payments, or on customers who themselves clear in dollars is exposed to the chilling effect even before it is directly named. Price that fragility. A useful framing is to separate *direct* designation risk (the firm itself gets listed) from *transmitted* risk (a key customer, supplier, lender, or correspondent bank gets listed, or simply de-risks the firm out of caution). The transmitted risk is the one investors miss: a perfectly clean company can have its dollar lifelines cut because its bankers decide it sits too close to a sanctioned counterparty. The firms most exposed are those with concentrated dependence on a single dollar-clearing relationship, those operating in or near a sanctioned jurisdiction, and those in industries (energy, defense, dual-use technology, finance) that draw the most sanctions attention. Build that exposure map for every geopolitically-sensitive position you hold.

**3. Price the designation tail.** Treat a possible designation as a **binary event** and size it. If you judge the probability of designation over your horizon at *p*, and a designation takes the asset to roughly zero for you, then the expected loss from the tail is about *p* × (your position). A position you'd happily hold at *p* = 1% is a very different animal at *p* = 20% (e.g., a firm already under sectoral sanctions in a fast-escalating conflict). Demand extra yield/discount to hold the tail, and cap position size accordingly. This binary-event framing is the same machinery used to trade [regulatory and binary events](/blog/trading/law-and-geopolitics/how-to-trade-a-regulatory-event) generally.

**4. Watch the escalation ladder and the licenses.** Designations rarely come from nowhere. The sequence is usually: rhetoric → diplomatic warning → targeted individual designations → sectoral/SSI sanctions → broad blocking designations → comprehensive program (and sometimes a reserve freeze or SWIFT cutoff at the extreme). Each rung raises *p*, and the rungs are observable: official statements, draft legislation (CAATSA-style bills signal intent), allied coordination (when the EU, UK, and US move together the program tends to be broad and durable), and the tempo of new listings. The discipline is to *re-price your designation probability each time the ladder advances a rung*, and to act before the rung most market participants are watching. By the time the blocking designation is announced, the move is over — the edge is in reading the rungs that precede it. And when a designation does land, the *single most important document for a holder* is the **wind-down general license** — its existence, scope, and expiry determine whether you can exit to a non-US buyer or are simply frozen. Read it within the hour, because the wind-down window is where the exit liquidity is, and it closes.

**5. Position the second-order trades.** Designations move more than the target. Defense and aerospace equities tend to rally on escalation (rising geopolitical risk → higher defense budgets); energy reroutes create discount/premium spreads between sanctioned and compliant grades; safe havens (gold, the dollar itself, Treasuries) bid on the geopolitical shock. The macro and cross-asset transmission of these shocks is the subject of [macro trading](/blog/trading/macro-trading) — sanctions are one of the cleanest "rule changes the price of an asset class" catalysts there is. The reason the second-order trades are often *better* than betting on the target directly is liquidity and capacity: the frozen target is, by definition, untradeable, so you cannot express a view on it after the fact, whereas the defense ETF, the energy producer, the gold position, and the dollar are all deep, liquid markets you can size and exit. The discipline is to pre-build the "geopolitical escalation" basket — long defense, long energy producers outside the conflict, long havens, short the most sanctions-exposed names — so that when the escalation ladder advances you are repositioning an existing framework rather than scrambling to react. The target going to zero is the headline; the basket around it is where most of the realizable return lives.

**What invalidates the view.** Be honest about when you're wrong. A designation-tail short or avoid is *invalidated* if (a) a broad, durable **general license** is issued that restores tradeability, (b) the target is **delisted** (sanctions are removed — it happens, often as part of a diplomatic deal), or (c) the political will behind the program evaporates (a change in administration or a peace settlement). Conversely, a "this asset is safe" thesis is invalidated the moment the escalation ladder advances a rung — sectoral sanctions on a sector you're in, or a designation of a major counterparty. Sanctions risk is *path-dependent and political*; size it as a tail you cannot perfectly hedge, not a risk you can model away.

The single mental model to carry away is this: **a designation does not change what an asset is worth — it changes whether you are allowed to own it.** Ordinary analysis asks "what are the cash flows and what discount rate do they deserve?" Sanctions analysis asks a prior question — "will I retain the legal right to hold and trade this at all?" — and when the answer turns negative, the cash-flow analysis becomes irrelevant, because a frozen asset is worth approximately nothing to you no matter how strong its fundamentals. That is why sanctions sit at the top of the law-policy-markets transmission chain: they are the purest case of a *rule change* overriding *fundamentals*, the law reaching past the balance sheet to revoke the right itself. Build the habit of asking the legal-capacity question first for any geopolitically exposed position, size the tail conservatively because you cannot stop-loss your way out of it, and use the wind-down window if you're caught. The same legal toolkit that freezes assets abroad also constrains money at home — the broader family of restrictions on capital movement is covered in [capital controls and the legal limits on money flows](/blog/trading/law-and-geopolitics/capital-controls-and-the-legal-limits-on-money-flows).

## Further reading & cross-links

Within this series:

- [The 2022 Russia sanctions and the weaponization of finance](/blog/trading/law-and-geopolitics/the-2022-russia-sanctions-and-the-weaponization-of-finance) — the case study: SWIFT cutoff, the \$300bn reserve freeze, the oil price cap, and the de-dollarization debate it triggered.
- [Sanctions evasion, enforcement, and compliance risk](/blog/trading/law-and-geopolitics/sanctions-evasion-enforcement-and-compliance-risk) — the defender's lens: how evasion actually works, record OFAC fines, and the investor's exposure to compliance blowups.
- [Capital controls and the legal limits on money flows](/blog/trading/law-and-geopolitics/capital-controls-and-the-legal-limits-on-money-flows) — the broader legal toolkit for restricting money movement.
- [How law moves markets: the transmission chain](/blog/trading/law-and-geopolitics/how-law-moves-markets-the-transmission-chain) — the spine: how any rule change discounts into prices before it bites.

Cross-asset and mechanism:

- [The petrodollar and dollar dominance](/blog/trading/finance/petrodollar-and-dollar-dominance) — why the world clears in dollars, the precondition that makes the chokepoint exist.
- [Macro trading](/blog/trading/macro-trading) — how geopolitical and policy shocks transmit through liquidity, rates, and asset classes.
