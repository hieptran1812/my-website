---
title: "CFIUS and national-security investment screening: when the state vetoes a deal"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "How a US government committee can block or unwind a cross-border deal that every shareholder and regulator approved, and how that veto power prices into merger-arb spreads, forced divestitures, and China-exposed names."
tags: ["regulation", "geopolitics", "cfius", "national-security", "merger-arbitrage", "foreign-investment", "outbound-screening", "deal-risk", "m-and-a", "china"]
category: "trading"
subcategory: "Law & Geopolitics"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — A national-security review can kill a cross-border deal that every shareholder, every board, and every antitrust regulator has already approved.
>
> - CFIUS — the Committee on Foreign Investment in the United States — is a Treasury-led interagency body that screens foreign investment for security risk and can force changes, force a sale, or recommend the President block the deal outright.
> - The mandate is far broader than people assume: since FIRRMA (2018) it covers not just controlling acquisitions but minority stakes in "TID" businesses — critical **T**echnology, critical **I**nfrastructure, and sensitive personal **D**ata — plus real estate near sensitive sites.
> - The TikTok/ByteDance forced-divestiture saga is the template, and a new **outbound** regime now screens US money flowing *into* Chinese tech.
> - The one number to remember: a foreign-buyer deal can trade **5 to 10 percentage points wider** on the merger-arb spread than a domestic comp at the same offer price — that gap *is* the market's price for "the state might say no."

In March 2018, a US President signed an executive order that vaporized a \$117 billion deal in a single afternoon. Broadcom — then domiciled in Singapore — was trying to take over Qualcomm, the San Diego chipmaker at the heart of America's 5G ambitions. The two boards were still fighting over price, but the deal was alive and the arbitrage desks were positioned for it to close. Then the order came down: the transaction was "prohibited," along with "any substantially equivalent merger." No court ruled. No antitrust authority objected. A committee most investors had never heard of had concluded that letting Broadcom slash Qualcomm's R&D would hand the 5G lead to Huawei — and that was enough.

That committee is CFIUS, and its veto is one of the strangest powers in modern markets. It does not care whether the price is fair. It does not care whether shareholders want the money. It does not balance competition against consumer welfare the way an antitrust regulator does. It asks one question — *does this transaction create an unacceptable risk to US national security?* — and if the answer is yes, a private deal between willing parties simply does not happen.

For an investor, this turns a piece of administrative law into a pricing factor. The moment a buyer is foreign — and especially when "who is the buyer's government" is China — every deal carries a tail you have to size: the tail where commercial logic loses to security logic. This post builds CFIUS from zero, walks the review clock, dissects the landmark cases, and then does the only thing that matters for a trader: shows how that veto power widens spreads, craters positions, and discounts whole businesses long before any rule actually bites.

![Approval flow where a foreign buyer, board, shareholders, and antitrust all approve but CFIUS can still block or force divestiture](/imgs/blogs/cfius-and-national-security-investment-screening-1.png)

## Foundations: what CFIUS is and where its power comes from

Start with the everyday version. Suppose you own a small machine shop and you want to sell it. You find a buyer, you agree on a price, you both sign, your accountant blesses it, and the local business-licensing office has no objection. In almost every transaction on earth, that is the end of the story — the deal is done. Now add one wrinkle: your machine shop happens to make a part that goes into military drones, and your buyer is a fund controlled by a foreign government that the US considers an adversary. Suddenly there is an invisible thirteenth party at the table who never signed anything and who can still say no. That party is the national-security state.

CFIUS is the formal name for that party in the United States. The letters stand for the **Committee on Foreign Investment in the United States**, and the plain-English description is: a standing committee of US government agencies that reviews foreign investments into American businesses for the risk they pose to national security, and that has the legal teeth to change, condition, unwind, or kill those investments.

### Who actually sits on the committee

CFIUS is **chaired by the Secretary of the Treasury**, which tells you something important: the body lives at the intersection of money and security, and Treasury — the agency that thinks about capital flows — runs the meeting. But it is genuinely interagency. The voting members include the Departments of Justice, Homeland Security, Commerce, Defense, State, and Energy, plus the Office of the US Trade Representative and the Office of Science and Technology Policy. Intelligence agencies and the Department of Labor participate without a vote. Why the crowd? Because "national security" is not one agency's expertise. Whether a Chinese acquisition of a port-logistics firm is dangerous depends on what Defense thinks about the port, what Commerce thinks about the technology, what Justice thinks about the data, and what the intelligence community thinks about the buyer. The committee is the room where those judgments are pooled.

### The legal authority

This is not a body that invents its own power. It rests on a statute. The original grant is the **Defense Production Act of 1950**, a Cold War law that lets the government steer the economy in the interest of national defense. A 1988 amendment — **Section 721**, commonly called the **Exon-Florio amendment** — gave the President the explicit authority to suspend or prohibit a foreign acquisition that "threatens to impair the national security." That presidential block power is the nuclear option, and we will come back to how rarely it is actually used.

The modern shape of CFIUS, though, comes from **FIRRMA — the Foreign Investment Risk Review Modernization Act of 2018**. Before FIRRMA, CFIUS could only review deals where a foreign person gained *control* of a US business. FIRRMA blew the doors open. It extended jurisdiction to certain *non-controlling* investments, created the new "TID" categories, added real-estate transactions near sensitive sites, and — crucially — introduced *mandatory* filings for some deals where previously every filing was voluntary. FIRRMA is the reason a venture fund taking a 5 percent stake in an AI startup now sometimes has to think about CFIUS at all.

### What "covered transaction" means

A deal only matters to CFIUS if it is a **covered transaction** — a term of art. There are two main doors in.

The first door is **control**. Any transaction that could result in a foreign person controlling a US business is covered. "Control" is defined functionally, not by a magic percentage: it is the power to determine important matters — board seats, veto rights over major decisions, the ability to direct strategy. A 40 percent stake with two board seats and a veto over the budget is "control"; a passive 60 percent stake with no governance rights might not be.

Why is "control" defined so loosely? Because a bright-line percentage would be trivially easy to game. If the rule were "50% or more," a foreign buyer would take 49% and a contract that gave it everything but the title. So Congress and the regulations made control a *substance-over-form* test. The factors CFIUS weighs include: the power to appoint or remove directors and senior officers; veto rights over the budget, strategic direction, or major contracts; rights over the sale of assets or the dissolution of the company; and access to the company's technical secrets. A foreign investor with a 13% stake, two board seats, and a veto over R&D spending can be deemed to have "control"; a passive index fund with a larger economic stake and no governance footprint generally does not. For a deal-risk analyst, the practical takeaway is that you cannot clear a deal of CFIUS exposure just by pointing at a sub-50% stake — you have to read the *governance terms*.

The second door, new since FIRRMA, is the **TID business**. Even a *non-controlling* investment is covered if the target is a TID business and the investment gives the foreign investor access to material non-public technical information, a board seat, or substantive decision-making rights. TID stands for:

- **T — critical Technology.** Items already controlled for export (think advanced semiconductors, encryption, certain biotech, aerospace), plus a category of "emerging and foundational technologies" that the government can expand over time. The list is deliberately a moving target: as a technology becomes strategically important, it can be folded in. This is why a venture investment that looked CFIUS-irrelevant in 2017 — a stake in an early AI or quantum startup — can be squarely covered today.
- **I — critical Infrastructure.** Systems whose incapacity would debilitate security or the economy: power grids, telecoms backbones, ports and terminals, water systems, financial-market utilities like clearinghouses. The test is not "is this a big company?" but "would its compromise cascade into national harm?"
- **D — sensitive personal Data.** Genetic data, precise geolocation, health records, financial records, and other datasets — particularly when they cover large numbers of Americans or specific sensitive populations (military personnel, government employees). The fear is concrete: a foreign intelligence service that holds the location histories and health profiles of millions of citizens can map who is vulnerable, who can be coerced, and who works where. Data is treated as a strategic asset in its own right.

There is a third, narrower door for **real estate**: purchases or leases of land near military bases, ports, airfields, or other sensitive government sites, even with no operating business attached. The textbook concern is a foreign-owned farm or facility sitting next to an air base, positioned to surveil flight operations. After several high-profile land-purchase controversies, this door has been widened to cover proximity to a growing list of sensitive sites.

![Decision tree showing the two doors into CFIUS jurisdiction, control of a business or a non-controlling stake in a TID business](/imgs/blogs/cfius-and-national-security-investment-screening-2.png)

### Voluntary, but risky

Here is the feature that makes CFIUS so strange to trade around: for most deals, **filing is voluntary**. There is no general legal requirement to notify the committee before you close. You can simply do the deal.

The catch is that CFIUS has **indefinite look-back power**. If you skip the filing and CFIUS later decides the transaction was a problem, it can review it *after the fact* — years later — and force you to unwind it. So "voluntary" is voluntary the way buying insurance is voluntary. Sophisticated parties file precisely to get a **"safe harbor"**: once CFIUS reviews and clears a deal, it generally cannot reopen it. You file not because you must, but because a clean clearance is the only thing that makes the deal permanent. (FIRRMA added a small set of *mandatory* filings — chiefly for certain TID deals and for foreign-government-controlled buyers — but the bulk of the system still runs on this "voluntary-but-you'd-be-crazy-not-to" logic.)

### Mitigation versus a block

When CFIUS finds a risk, it has a ladder of responses, and the top rung is far rarer than the headlines suggest.

The most common non-clearance outcome is a **mitigation agreement** — a contract between the parties and the government that lets the deal proceed *with conditions*. Typical terms: carve out the sensitive US assets from the sale, install a US-citizen security officer, ring-fence data so the foreign parent can't touch it, give the government audit rights, or require that certain facilities stay on US soil. Mitigation is CFIUS saying "yes, but on a leash."

Above mitigation is the deal being **abandoned** — the parties walk away voluntarily, usually because CFIUS has signaled it won't clear and they'd rather kill the deal quietly than suffer a public presidential prohibition.

At the very top is the **presidential block** or a **forced divestiture** — the order to stop the deal, or to unwind one already done. This is the power that prohibited Broadcom-Qualcomm. It is used a handful of times per decade, but its *shadow* is everywhere, because the threat of it is what drives the mitigation-and-abandonment that happens far more often.

> [!note]
> **The mental shortcut.** CFIUS is not an antitrust regulator with a different rulebook. Antitrust asks "will this hurt competition and consumers?" CFIUS asks "will this hurt national security?" A deal can sail through antitrust and die at CFIUS, or vice versa. They are *two independent veto points*, and a foreign-buyer deal has to clear both.

To see how different the two reviews are, line them up. Antitrust review (the Hart-Scott-Rodino process run by the DOJ and FTC, covered in [antitrust-101-sherman-clayton-and-merger-review](/blog/trading/law-and-geopolitics/antitrust-101-sherman-clayton-and-merger-review)) cares about *market structure*: does the combined firm get too much pricing power, measured by concentration metrics like the Herfindahl-Hirschman Index? A merger of two giants in the same market triggers antitrust scrutiny *regardless of who owns them* — a deal between two US companies can be blocked on competition grounds, and a deal between two foreign companies usually isn't reviewed by CFIUS at all if no US business is involved. CFIUS is the mirror image: it does not care about concentration or consumer prices in the slightest. It cares about *who the buyer answers to* and *what the target touches*. A monopoly-creating merger between two domestic firms is a pure antitrust problem and a CFIUS non-event; a tiny, competition-irrelevant minority investment by a state-linked foreign fund into a chip startup is a pure CFIUS problem and an antitrust non-event. The two regimes use different inputs, ask different questions, and have independent kill switches — which is exactly why a foreign-buyer deal carries *two* tails an analyst must price, not one.

### Why this matters more now than a decade ago

It is worth being explicit about the regime shift, because the intensity of CFIUS is not a constant — it is a policy variable that has ratcheted in one direction. For most of its history, CFIUS was a sleepy, control-only committee that reviewed a modest number of deals and blocked almost none. Three forces changed that. First, **technology became security**: as semiconductors, AI, biotech, and data moved to the center of military and economic power, the set of "sensitive" businesses expanded enormously. Second, **great-power competition returned**: the bipartisan consensus that China is a strategic rival turned foreign investment from a welcome inflow into a vector of risk. Third, **FIRRMA gave the committee the legal tools** to act on the first two — minority-stake jurisdiction, the TID categories, mandatory filings, and a bigger budget and staff. The net effect is that the *probability* of any given foreign-buyer deal drawing scrutiny has risen structurally, and the *breadth* of what counts as sensitive keeps expanding. An investor pricing CFIUS risk off a pre-2018 base rate is using a stale model.

## How the review clock actually runs

The thing that makes CFIUS *tradable* is that it runs on a statutory clock. A trader who knows the clock can map the catalyst calendar of a deal almost the way you'd map an earnings date.

A filing comes in one of two forms. The light version is a **declaration** — a short, roughly five-page form that CFIUS has 30 days to assess and respond to. The full version is a **written notice** — a heavy, detailed filing that triggers the main statutory clock.

That main clock has two phases plus a presidential tail:

- **The 45-day review.** Once CFIUS accepts a full notice, it has 45 days to conduct an initial review. Many clean deals clear right here.
- **The 45-day investigation.** If the review surfaces unresolved national-security concerns, CFIUS opens a formal investigation — another 45 days. This is where mitigation gets negotiated. Add the phases and you are at roughly 90 days.
- **The 15-day presidential decision.** If the committee cannot resolve its concerns and cannot reach a mitigation deal, it refers the transaction to the President, who has 15 days to act — block it, or let it proceed.

So the headline number is **about 105 days** from a clean full notice to a final answer, but the real-world timeline is longer. Parties often hold pre-filing consultations for weeks. And there is an escape valve the committee uses constantly: it can ask the parties to **withdraw and refile**, resetting the clock. A deal that "is taking nine months at CFIUS" is usually a deal that has refiled once or twice while mitigation gets hammered out.

![Timeline of the CFIUS review clock from filing through the forty-five day review, forty-five day investigation, and the presidential decision](/imgs/blogs/cfius-and-national-security-investment-screening-3.png)

For the trader, the clock means the risk is *time-distributed*, not a single binary date. A domestic merger-arb position lives or dies mostly on the antitrust calendar. A foreign-buyer position has a *second* clock running underneath, and the spread should fatten as the deal approaches the dangerous phases — the opening of an investigation, the rumor of a refile, the referral to the President.

#### Worked example: the implied probability of a CFIUS block

The whole point of merger arbitrage is that the spread between the market price and the deal price encodes a probability. Let me make that concrete.

A US target, NatSecCo, agrees to be acquired for **\$100.00 per share** in cash. There are two universes. In universe A, the buyer is a US private-equity firm. In universe B, the buyer is a foreign state-linked fund, and NatSecCo makes radio-frequency components used in defense systems — a textbook TID business.

In universe A, the stock trades at **\$98.50**. The spread is \$100.00 − \$98.50 = \$1.50, or **1.5%**. That spread compensates the arbitrageur for the ordinary risk of any deal breaking (financing falls through, a material-adverse-change clause, garden-variety regulatory delay) plus the time value of money to close.

In universe B, the *same* deal, *same* price, trades at **\$92.00**. The spread is \$8.00, or **8.7%**. The extra spread is \$8.00 − \$1.50 = **\$6.50 per share** of pure CFIUS risk premium.

To turn that into a probability, you need the downside. If the deal breaks on a CFIUS block, where does the stock fall back to? Suppose NatSecCo's undisturbed, no-deal price is **\$70.00**. Then the loss on a break is \$100.00 − \$70.00 = **\$30.00 per share**.

A rough single-period merger-arb identity: the spread ≈ probability-of-break × loss-on-break (ignoring upside and carry for clarity). The *extra* spread in universe B is the *extra* probability of break times the loss:

\$6.50 = P(extra break) × \$30.00, so **P(extra break) ≈ 6.50 / 30.00 ≈ 21.7%**.

In words: the market is pricing roughly a **one-in-five extra chance** that this otherwise-identical deal dies *specifically because the buyer is foreign and the business is sensitive*. That is the CFIUS factor, made into a number you can trade against. If your own read says CFIUS is more likely to clear-with-mitigation than the market fears, you buy the wider spread; if you think a block is more likely than 22%, you stay away or short the spread.

![Bar comparison of a domestic buyer trading at a narrow spread versus a foreign buyer at a wide spread with the implied break probability](/imgs/blogs/cfius-and-national-security-investment-screening-4.png)

## The outcome ladder, and why blocks are rare but devastating

It is tempting to model CFIUS as a coin flip — clear or block. It is not. It is a ladder, and understanding the rungs is what separates a good deal-risk analyst from a headline-reader.

The bottom rung is **no filing needed**: the deal isn't a covered transaction (a foreign buyer of a US restaurant chain with no tech, no critical infrastructure, no sensitive data, far from any base). The next rung is **clearance in the 45-day review** — the committee looks and finds nothing. Above that, **clearance after a full investigation** — concerns were raised and resolved. Above *that*, **clearance with a mitigation agreement** — the deal lives but on conditions. Then **abandonment** — the parties walk before a formal block. And finally, at the very top, the **presidential block or forced divestiture**.

Why does the distribution matter so much? Because the rungs have wildly different payoffs for a position. Mitigation usually lets the deal close near the original price (maybe with a small haircut for the carve-outs), so a mitigated deal is close to a win for the arbitrageur. A block is a total loss of the spread plus the crater back to the undisturbed price. The expected value of a foreign-buyer position is dominated by the *shape* of this distribution, not by any single outcome.

![Outcome ladder contrasting heavier interventions like a presidential block against lighter outcomes like clearing in review](/imgs/blogs/cfius-and-national-security-investment-screening-5.png)

Empirically, outright presidential blocks are rare. Across the modern era the President has formally prohibited only a handful of transactions. Far more deals are *mitigated* — CFIUS reports that a meaningful share of reviewed notices end in mitigation agreements — and a larger number are quietly **abandoned** when the committee signals it won't clear. The lesson: the dangerous outcome for a position is not usually the dramatic presidential order; it is the quiet abandonment, which produces exactly the same crater in the stock but with less warning, because the parties pull the deal rather than wait for a public block.

#### Worked example: the deal-break crater on a CFIUS position

Suppose you run a \$50 million merger-arb book and you put **\$5 million** into the universe-B NatSecCo deal at **\$92.00**, betting CFIUS clears it. You buy 54,348 shares (\$5,000,000 / \$92.00).

If the deal clears at \$100.00, you make \$8.00 per share: 54,348 × \$8.00 = **\$434,784**, an 8.7% gross return on the position over, say, six months — annualizing to a juicy ~17%. That is why the wide spread is tempting.

Now the bad branch. CFIUS refers the deal to the President, who blocks it. NatSecCo falls back to its undisturbed \$70.00. Your loss is \$92.00 − \$70.00 = \$22.00 per share: 54,348 × \$22.00 = **\$1,195,656**. You lose **\$1.2 million** — about **24% of the \$5 million position**, and 2.4% of the entire book, on a single name, in a single afternoon when the order drops.

Notice the asymmetry: you risked \$22.00 to make \$8.00 — a 2.75-to-1 payoff *against* you. For the trade to have positive expected value, your probability of a clear has to be high enough to overcome that ratio. Break-even is when P(clear) × \$8.00 = P(break) × \$22.00, i.e. P(clear) / P(break) = 22/8 = 2.75, so **P(clear) must exceed about 73%**. If you can't get comfortable that CFIUS clears at least three times out of four, the wide spread is a trap, not a gift.

## The anatomy of a mitigation agreement

Because mitigation — not the dramatic block — is the most common non-clearance outcome, a serious analyst should understand what it actually looks like. A mitigation agreement is a binding contract between the deal parties and the US government that lets the transaction close *only if* the parties accept ongoing conditions designed to neutralize the security risk. It is the regulatory equivalent of "you can keep the dog, but it has to be muzzled, leashed, and you'll let an inspector check the leash whenever they want."

The conditions fall into a few families:

- **Structural carve-outs.** The sensitive piece of the business is excluded from the foreign buyer's reach. A foreign acquirer of a conglomerate might be required to divest the one US subsidiary that holds a classified defense contract before the rest of the deal closes. The carve-out is the cleanest fix because it removes the risk rather than managing it.
- **Governance ring-fences.** A "Special Security Agreement" or "proxy agreement" inserts a layer of US-citizen directors, a government-approved security officer, and firewalls so the foreign parent cannot access sensitive operations, contracts, or data. The foreign owner gets the economics but is walled off from the controls.
- **Data and technology controls.** The classic example is data localization — sensitive US-person data must stay on US soil, be managed by US personnel, and be inaccessible to the foreign parent. (TikTok's "Project Texas" was an attempt at exactly this kind of mitigation at enormous scale; the fact that it ultimately did not satisfy the government shows that mitigation can *fail* on the most sensitive deals.)
- **Audit and monitoring rights.** The government gets the right to inspect, audit, and verify compliance — often with a third-party monitor paid for by the company. Mitigation is not a one-time fix; it is an ongoing supervisory relationship that can last for the life of the investment.

For the investor, mitigation matters because it has a *cost* and a *failure mode*. The cost is real: carve-outs shrink the asset the buyer is paying for, and compliance machinery is expensive to run. So a deal that closes "with mitigation" often closes at a slightly lower effective value than the headline price — the buyer paid for a muzzled dog. The failure mode is that mitigation can break down later: if the company is caught violating the agreement, the government can re-open the matter, impose penalties, or in the extreme force divestiture. A mitigated deal is *mostly* a win for an arbitrage position, but it is not a clean clearance.

#### Worked example: the expected value of a foreign-buyer position across the outcome ladder

This is where the ladder becomes a single number. Take the universe-B NatSecCo deal trading at **\$92.00** on a **\$100.00** offer, with an undisturbed price of **\$70.00**. Instead of a two-outcome coin, build the full distribution from the case pattern. Suppose your research gives:

- **Clean clearance**, deal closes at \$100.00 — probability **40%**.
- **Clearance with mitigation**, deal closes at an effective **\$97.00** (a \$3 haircut for carve-outs) — probability **35%**.
- **Forced abandonment or block**, stock falls to \$70.00 — probability **25%**.

The probability-weighted exit price is:

0.40 × \$100.00 + 0.35 × \$97.00 + 0.25 × \$70.00 = \$40.00 + \$33.95 + \$17.50 = **\$91.45**.

Against the \$92.00 you'd pay, the expected exit of \$91.45 means a *negative* expected value of about **−\$0.55 per share** — the deal is slightly *too tight* for your distribution, and you'd pass. Now flip one input: if your research says the block probability is only **15%** (and clean clearance **50%**), the weighted exit becomes 0.50 × \$100 + 0.35 × \$97 + 0.15 × \$70 = \$50.00 + \$33.95 + \$10.50 = **\$94.45**, a healthy **+\$2.45 per share** of edge over the \$92.00 price. The entire trade lives or dies on that 10-point swing in your block probability — which is precisely the judgment the case pattern is for. The lesson: you are not betting on "does CFIUS clear?"; you are betting on the *shape of the whole ladder*, and small shifts in the tail rung dominate the answer.

## Landmark cases: the pattern behind the headlines

Cases are where the abstract triggers become concrete. Lay the big ones side by side and a pattern jumps out: **critical technology or sensitive data, plus a buyer linked to a strategic rival, equals intervention.**

![Matrix of landmark CFIUS cases showing each deal, its sector, the security concern, and the outcome](/imgs/blogs/cfius-and-national-security-investment-screening-6.png)

**Broadcom–Qualcomm (2018, blocked).** The cleanest illustration of the new logic. The concern was not that Broadcom was Chinese — it wasn't — but that a hostile takeover would gut Qualcomm's long-horizon R&D and cede the global 5G standards race to Huawei. CFIUS intervened *before the deal was even signed*, using an unusual interim order, and the President prohibited it. The lesson: CFIUS can act on a *competitiveness-as-security* theory, and it can act early.

**Lattice Semiconductor–Canyon Bridge (2017, blocked).** Canyon Bridge was a private-equity firm funded in part by Chinese state capital. Lattice makes programmable chips. The President blocked the \$1.3 billion deal. The lesson: the *source of the money* matters as much as the identity of the front buyer — CFIUS looks through the structure to the ultimate funding.

**MoneyGram–Ant Financial (2018, abandoned).** Jack Ma's Ant Group tried to buy the US money-transfer firm MoneyGram for about \$1.2 billion. CFIUS wouldn't clear it; the concern was the security of Americans' financial data and the payments rails. The parties abandoned the deal and Ant paid a \$30 million break fee. The lesson: **data is a national-security asset**, and a payments network is critical financial infrastructure.

**Grindr–Kunlun (2019–20, forced divestiture).** This is the one that scared every fund holding a US consumer-app business. Kunlun, a Chinese gaming company, had *already bought* Grindr, the LGBTQ dating app. CFIUS reviewed it *after the fact* and concluded that the app's trove of sensitive personal data — locations, health status, private messages — in the hands of a Chinese owner was an unacceptable risk. It forced Kunlun to **sell**. The lesson: the look-back power is real, sensitive personal data is a first-class trigger, and a "completed" deal is not necessarily a closed chapter.

**TikTok–ByteDance (the divestiture template).** TikTok is the defining CFIUS story of the decade, and it shows how the tool escalates when the committee can't get comfortable. The roots go back to ByteDance's 2017 acquisition of Musical.ly, the app it merged into TikTok — a deal that was *not* filed with CFIUS at the time. Years later, with TikTok grown into a platform used by well over 100 million Americans, CFIUS opened a review of that old acquisition under its look-back power, and the security concerns crystallized around two questions: could the Chinese government compel ByteDance to hand over American users' data, and could it manipulate the recommendation algorithm to shape what Americans see?

CFIUS spent years trying to negotiate a mitigation deal — the "Project Texas" plan, under which US user data would be stored on US servers run by a US cloud provider, with American oversight of the algorithm. It was mitigation at an unprecedented scale, reportedly costing ByteDance well over a billion dollars to build. And it *still* did not satisfy the government, which is the single most important lesson of the whole saga: **on the most sensitive deals, even an enormous, good-faith mitigation effort can fail.** When the CFIUS track stalled, the action moved to Congress, which in 2024 passed a law requiring ByteDance to **divest** TikTok's US operations or see the app removed from US app stores. Whether you view it as a CFIUS outcome or a legislative one, the *mechanism* is identical to Grindr at national scale: a foreign-owned app holding the data and the algorithmic attention of a huge slice of Americans is treated as a security problem, and the remedy is a forced sale. This **forced-sale-or-ban template** is now the reference point for every deal involving Chinese ownership of a consumer-facing data business — and it is why any investor underwriting a Chinese-owned US app should assume the *tail outcome is a forced divestiture*, not a quiet clearance.

**Nippon Steel–US Steel (2023–25, blocked then litigated).** Japan is a treaty ally, which makes this case the important *exception* to "CFIUS only blocks adversaries." Nippon Steel of Japan agreed to buy US Steel for roughly \$14.9 billion. CFIUS reviewed it on supply-chain-security grounds — domestic steel capacity is treated as strategic — and the President moved to block it, prompting litigation and a drawn-out fight that eventually reshaped the deal's structure. The lesson, which we'll hammer in the misconceptions section: **even an ally's buyer can trip the wire** when the target sits in a politically sensitive strategic industry.

The through-line: in every case, the analyst could have flagged the risk *in advance* from two inputs — the sensitivity of the target (tech, infrastructure, data, strategic industry) and the buyer's government linkage. That is exactly the screen a deal-risk desk should run.

## Common misconceptions

#### "CFIUS only blocks Chinese deals."

This is the most expensive myth, because it lulls analysts into treating any non-Chinese buyer as CFIUS-safe. China is unambiguously the dominant focus — in recent reporting years, Chinese acquirers have accounted for one of the largest national shares of covered transactions, and the Chinese share of *blocks and forced divestitures* is higher still. But the wire is not labeled "China only."

The Nippon Steel–US Steel block is the proof: Japan is a treaty ally, and the deal was still moved to block on strategic-industry grounds. CFIUS reviews buyers from dozens of countries every year; it has scrutinized deals involving Gulf sovereign-wealth funds, European telecoms, and others. FIRRMA even created an **"excepted foreign state"** carve-out for close allies (Australia, Canada, the UK, New Zealand) — but it is partial, conditional, and does *not* exempt the most sensitive deals. **The correct mental model:** China is the highest-probability trigger, but the right question is never "is the buyer Chinese?" — it is "does this combination of sensitive target and foreign government linkage create a security story someone in Washington can tell?" A Japanese steel deal had such a story; that was enough.

#### "A private deal between willing parties doesn't need government sign-off."

In the ordinary case, true — most M&A closes with no national-security review at all. But for a covered transaction, this myth gets people unwound *years later*. The Grindr case is the cautionary tale: Kunlun completed the acquisition, integrated it, and operated it — and CFIUS still reached back, reviewed it, and forced a sale. There was no statute that *required* Kunlun to file before closing; the company simply bet it didn't need to, and lost.

The number that matters: CFIUS's look-back power has **no statute of limitations** for unfiled deals. A clean closing is *not* a safe harbor; only a CFIUS clearance is. For an investor, this means a portfolio company sitting on an *unreviewed* foreign acquisition of a sensitive business carries a latent liability that can detonate at any time — and the market will eventually price that overhang even before it detonates.

#### "National-security review is rare, so I can ignore it."

The blocks are rare; the *reviews* are not, and the *chilling effect* is enormous. CFIUS now processes hundreds of covered transactions per year — well over 400 filings annually in recent reporting — and a substantial fraction end in mitigation agreements rather than clean clearances. The committee's footprint expanded sharply after FIRRMA in 2018; the count of declarations and notices has climbed, and entire categories of deals (minority TID investments, real estate near bases) that were invisible to CFIUS before 2018 now routinely cross its desk.

More important than the raw count is the **deterrence**. The deals that *never get announced* because the bankers told the client "a Chinese buyer will never clear CFIUS, don't bother" don't show up in any statistic. The chilling effect on Chinese direct investment into US tech has been dramatic: after peaking in the mid-2010s at over \$45 billion in a single year, annual Chinese FDI into the United States collapsed by roughly **80–90%** from its highs to a low-single-digit-billions trickle, and the share going into sensitive technology fell even further. Ignoring CFIUS because blocks are rare is like ignoring building codes because collapses are rare — the rule is shaping every structure whether or not it ever visibly fails.

### The second-order effects: a different cost of capital for sensitive US assets

Step back and the deterrence has a price that shows up in valuations, not in any single deal. When a regime systematically removes a category of buyers from the market, it changes the *equilibrium* price of the assets that category used to bid for.

Think about who pays the strategic premium in M&A. Often it is a *foreign* strategic acquirer — a company that wants the target's technology, its market access, or its talent badly enough to outbid every financial buyer. When CFIUS effectively bars that foreign strategic from the auction for the most sensitive US assets, the auction loses its top bidder. The target is left with domestic strategics and financial sponsors, who pay synergy-and-returns math, not "I must own this for national reasons" math. The result is a structurally **lower takeout multiple** for the most sensitive US businesses than an otherwise-identical business in a non-sensitive sector would command. The same machine, the same cash flows, a smaller buyer pool — a lower clearing price.

This is the deepest way CFIUS "moves markets": not by the rare block that makes headlines, but by quietly compressing the exit multiple on an entire class of assets, every day, in deals that are never even attempted. An equity analyst valuing a sensitive-sector name on a takeout basis should haircut the assumed takeout multiple for exactly this reason — the foreign-premium tail has been clipped off the distribution.

## How it shows up in real markets

CFIUS doesn't announce itself with a flashing light. It shows up as three distinct market patterns, and learning to recognize them is the practical payoff of all the legal machinery above.

### Pattern 1 — the foreign-buyer deal trading at a stubbornly wide spread

The most common signature. A merger is announced; the antitrust analysis looks clean; the financing is solid; and yet the stock trades 5, 8, even 12 points below the deal price and *stays* there. When you see a spread that wide on a deal with no obvious antitrust problem, the first thing to check is the buyer's nationality and government linkage. A wide spread that antitrust can't explain is usually the market pricing CFIUS risk. The spread will often *widen further* on specific catalysts — the opening of a formal investigation, a leak that the parties have been asked to withdraw and refile, or political noise from members of Congress demanding the deal be blocked.

### Pattern 2 — the forced divestiture at a fire-sale price

When CFIUS forces a sale, the seller is a *distressed, deadline-driven* seller — the worst possible negotiating position. The asset has to be sold to an *approved* (usually domestic) buyer, within a *fixed window*, under threat of a total ban if no sale closes. Every bidder knows the seller has no walk-away option. The result is a predictable discount to what a willing seller in an open auction would get. Kunlun's forced sale of Grindr, and the structural pressure on any ByteDance-style divestiture, both carry this dynamic: the *fact* of the forced sale is value-destructive independent of the asset's quality.

#### Worked example: value destroyed in a forced divestiture

A foreign owner is forced by CFIUS to sell a US app it controls. In a normal, unpressured M&A auction, the business would fetch a fair value of, say, **\$4.0 billion** — call this the "willing-seller" price, set by competitive bidding and a patient timeline.

But this is a forced sale: a 180-day window, a ban if no deal closes, and only domestic buyers allowed (the pool of bidders just shrank). Forced, deadline-driven sales of distressed-but-good assets routinely clear at **25–40% discounts**. Say the asset sells at a 35% haircut: \$4.0 billion × (1 − 0.35) = **\$2.6 billion**.

The value destroyed *purely from the forced nature of the sale* is \$4.0 billion − \$2.6 billion = **\$1.4 billion**. That \$1.4 billion is not a market judgment that the app got worse — the app is identical. It is the cost of the state's veto landing on the seller's side of the table. For an investor *holding the foreign parent*, this is a direct, quantifiable hit to net asset value the moment a forced divestiture becomes likely; for an investor on the *buy side*, the same \$1.4 billion is the bargain on offer for being one of the few approved domestic bidders.

### Pattern 3 — the deal that dies before it's ever filed

The quietest pattern and the hardest to trade, because there is no announcement to react to. A foreign strategic acquirer would love to buy a US target; the bankers run a CFIUS pre-screen; the answer comes back "this will never clear"; and the approach is dropped before any LOI is signed. The target — which might have commanded a fat strategic premium from a global pool of buyers — is now effectively limited to *domestic* acquirers. That shrinks its buyer universe and caps the premium it can realistically fetch. You see this priced as a structurally lower takeout multiple for US assets in the most sensitive sectors (advanced semiconductors, defense electronics, sensitive-data platforms): the "foreign strategic premium" has been quietly zeroed out by the screening regime.

## The global spread: CFIUS has imitators everywhere

For two decades CFIUS was a relative oddity — a country willing to subordinate inbound capital to security review. That era is over. National-security investment screening has become a *global default*, and an investor in cross-border deals now has to map not one veto point but several.

![Grid comparing inbound screening regimes in the United States, United Kingdom, European Union, and China by who screens, the trigger, and the teeth](/imgs/blogs/cfius-and-national-security-investment-screening-7.png)

- **United Kingdom — the National Security and Investment Act (NSIA), in force since 2022.** A direct CFIUS analogue with sharper teeth in one respect: it has *mandatory* notification across 17 sensitive sectors (defense, AI, advanced materials, energy, communications, and more), and the government can call in, block, or unwind deals. Its retroactive call-in power reaches back years. Several Chinese-linked semiconductor and infrastructure deals have already been blocked or unwound under it.
- **European Union — a coordination framework, in force since 2020.** The EU regulation does *not* give Brussels a veto; foreign-investment screening remains a *member-state* competence. What the EU framework does is create an information-sharing and cooperation mechanism so member states (and the Commission) can flag concerns about a deal in one country that affects the whole bloc. The result is uneven: Germany, France, Italy, and others have their own robust national regimes, while some smaller members have light-touch ones. For a deal-risk analyst, this means the relevant veto sits at the *national* level, with EU-level coordination raising the odds that one member's concern becomes everyone's.
- **China — MOFCOM and the security-review regime.** China runs its own inbound national-security review (administered through MOFCOM, the Ministry of Commerce, alongside the NDRC) and, separately, a powerful merger-control regime that it has used as geopolitical leverage — most visibly by *slow-walking* approvals of US-led deals during periods of tension. A US company's global merger can be held hostage to Chinese sign-off if it does meaningful business in China.

The practical upshot: a large cross-border tech merger in 2026 might need to clear CFIUS in the US, the NSIA unit in the UK, multiple national screeners in the EU, *and* MOFCOM in China — any one of which can block or fatally delay it. The number of independent veto points has multiplied, and each one widens the deal's risk distribution.

## The new frontier: outbound-investment screening

Everything so far has been about *inbound* money — foreigners buying into US businesses. The newest and most consequential development reverses the arrow: **outbound screening**, which polices US money flowing *into* the businesses of a strategic rival.

The logic is a mirror image of CFIUS. The worry is no longer "a foreign adversary gains control of a sensitive US asset"; it is "US capital, talent, and know-how accelerate a foreign adversary's most dangerous capabilities." An executive order and implementing rules established a regime — effective in early 2025 — that restricts US persons from making certain investments into Chinese companies in three buckets: **advanced semiconductors and microelectronics, quantum information technologies, and certain artificial-intelligence systems** (especially those with military or surveillance applications). Some transactions are flatly **prohibited**; others require **notification**. This is a genuinely new instrument — the US has long restricted what you can *sell* to a rival (export controls); now it restricts what you can *fund* there.

For markets, outbound screening creates a slow-burn **overhang** on US firms with deep China exposure in these sectors — venture funds with China-tech portfolios, chip firms with Chinese JVs, AI companies with mainland research arms. The rule doesn't have to block a single transaction to depress value; it adds friction, freezes planned capital deployment, and forces a "China-discount" onto the affected business lines. This connects directly to the broader [export-controls-and-the-chip-war](/blog/trading/law-and-geopolitics/export-controls-and-the-chip-war) story: outbound screening is the *capital* leg of the same decoupling that export controls run on the *technology* leg.

#### Worked example: the outbound overhang on a China JV

A US chip-design firm owns a **50% stake in a China-based joint venture** that manufactures and sells into the mainland market. On a stand-alone basis — discounting its expected cash flows the way you would any business — that stake is worth **\$2.0 billion**. The market used to give the firm full credit for it.

Now outbound screening lands. Three things happen to that \$2.0 billion of value:

1. **Frozen capital.** The JV's growth plan assumed another **\$0.5 billion** of US capital injections to expand capacity. Outbound rules now prohibit or heavily restrict that funding, so the growth case evaporates and the present value of the lost expansion is roughly **−\$0.5 billion**.
2. **Multiple compression.** Even the *existing* JV cash flows now carry a regulatory tail — the risk of a forced wind-down, partner disputes, or a future tightening of the rules. Investors demand a higher discount rate, knocking another **−\$0.4 billion** off the value.
3. **Residual.** Stand-alone \$2.0bn − \$0.5bn frozen-growth − \$0.4bn multiple-compression = **\$1.1 billion** of value the market is actually willing to pay.

The **overhang is \$2.0bn − \$1.1bn = \$0.9 billion, about 45% of the stand-alone value** — and crucially, *not one transaction has been blocked yet*. The discount is the market pricing the *rule's existence*, exactly the way [regulatory-risk-as-an-asset-pricing-factor](/blog/trading/law-and-geopolitics/how-law-moves-markets-the-transmission-chain) predicts: the expected rule gets discounted into the price long before it bites. If you think the market has *over*-discounted (the rules turn out narrower than feared), the China-exposed name is a long; if you think the screening will tighten further, the overhang grows and the name is a short or an avoid.

![Waterfall of a China joint venture value from stand-alone two billion down to the one point one billion the market pays after the outbound overhang](/imgs/blogs/cfius-and-national-security-investment-screening-8.png)

## How to trade it — the playbook

Everything above converges on a handful of concrete moves. CFIUS is not a thing you trade directly; it is a *risk factor you price into deals and exposures*. Here is the operating manual.

### 1. Flag every foreign-buyer and sensitive-target deal for CFIUS risk

Build the screen into your deal-risk process. Two inputs decide almost everything:

- **The buyer's government linkage.** Is the acquirer foreign? Is it state-owned, state-funded, or from a country the US treats as a strategic rival? Look *through* the structure to the ultimate source of capital (the Lattice–Canyon Bridge lesson).
- **The target's sensitivity.** Is it a TID business — critical technology, critical infrastructure, sensitive personal data — or a politically strategic industry (steel, semiconductors, defense, ports)? Is it near a military base?

A deal that scores high on both axes is a CFIUS deal, full stop, even if no one has mentioned the committee yet. Price it as one.

### 2. Price the wider spread — and decompose it

When a foreign-buyer deal trades wide, separate the spread into its parts: ordinary deal-break risk (use a domestic comp), antitrust risk, and the *residual* CFIUS risk. Back out the implied probability of a CFIUS break using the spread-equals-probability-times-loss identity from the worked example. Then form your *own* view of that probability from the case pattern (sensitivity × buyer linkage) and trade the *gap* between the market's implied probability and yours. This is exactly the discipline of [merger-arbitrage-trading-regulatory-deal-risk](/blog/trading/law-and-geopolitics/merger-arbitrage-trading-regulatory-deal-risk) — CFIUS is simply one more veto point feeding the break probability.

### 3. Handicap the *outcome*, not just clear-versus-block

Because the outcome is a ladder, not a coin, your edge comes from estimating the *distribution*:

- **Clear with mitigation** (close to a win): likely when the concern is *specific and carve-out-able* — a single sensitive contract, a data set that can be ring-fenced. Bet on the deal.
- **Forced divestiture / abandonment** (a loss for the holder, a bargain for the approved buyer): likely when the sensitive asset is the *whole company* and can't be carved out (a dating app, a payments network).
- **Presidential block** (total loss of the spread): rare, reserved for the highest-profile, most-strategic deals, and usually telegraphed by political noise.

Weighting these correctly is the entire game. A deal where mitigation is plausible deserves a much tighter spread than one where the only options are clear or kill.

### 4. Map the outbound overhang onto China-exposed names

Outbound screening is a portfolio-level factor, not a deal event. Inventory your holdings for US firms with material China exposure in semiconductors, quantum, and AI. Estimate the *overhang* — the haircut for frozen capital plus multiple compression — the way the JV worked example did. Names where the market has *over*-discounted a manageable rule are longs; names where the rule is likely to *tighten* are avoids or shorts. This sits alongside the supply-chain decoupling trade in [supply-chain-geopolitics-reshoring-friendshoring-and-critical-minerals](/blog/trading/law-and-geopolitics/supply-chain-geopolitics-reshoring-friendshoring-and-critical-minerals).

### 5. Watch the catalysts — CFIUS runs on a clock

Set alerts for the events that move the spread: the announcement of a filing, the opening of a formal investigation, a "withdraw and refile" leak, congressional letters demanding a block, and the approach of the statutory deadline. Each is a discrete repricing moment. Because the clock is *known*, the catalysts are *schedulable* — treat them like the rulemaking calendar in any other regulatory trade.

### 6. Build the monitoring stack

CFIUS risk is readable in advance if you watch the right inputs, and the good news is that most of them are public. A practical monitoring stack for a deal-risk or event desk:

- **The deal docket.** Read the merger proxy and the merger agreement. The agreement will spell out which regulatory approvals are conditions to closing — if CFIUS clearance is a named condition, the parties themselves have told you they see the risk. The agreement's *reverse termination fee* (the break fee the buyer pays if the deal dies on regulatory grounds) is a direct signal of how much risk the buyer thinks it is carrying; an unusually large CFIUS-triggered break fee is a flashing light.
- **The buyer's ownership.** Trace the capital to its ultimate source — sovereign-wealth funds, state-owned enterprises, and funds with state limited partners all raise the risk score. The Lattice case is the reminder that the *front* buyer's nationality can hide the real exposure.
- **The political channel.** CFIUS itself is opaque, but the politicians around it are loud. Letters from members of Congress demanding a deal be blocked, hearings naming the transaction, and op-eds from former national-security officials are all *leading* indicators — they precede and pressure the formal review. A deal that has become a political talking point is a deal whose risk premium should widen.
- **The clock and the refiles.** Track the filing date and the statutory deadlines. A "withdraw and refile" — usually disclosed in an SEC filing or a press release — is a tell that mitigation is being negotiated and that the review is not clean. Each refile resets the clock and extends the period during which the spread stays wide.

The discipline mirrors the legal-and-geopolitical risk dashboard the rest of this series builds: turn an opaque government process into a set of observable, schedulable signals, and react to each one as a discrete repricing event.

### 7. Know what invalidates the view

A disciplined CFIUS thesis has explicit kill switches:

- **A safe-harbor clearance.** Once CFIUS clears a deal, the risk is *gone* — the spread should collapse to ordinary deal-break risk. If you were short the wide spread (betting on a block) and a clean clearance lands, you are wrong; cover.
- **A mitigation agreement on terms.** If you bet on a block but the parties strike a workable mitigation deal, the deal lives — exit.
- **A change in the political weather.** CFIUS is statutory but its *intensity* is political. A thaw in US–China relations, an "excepted foreign state" designation, or a narrowing of the outbound rules all *shrink* the risk premium; an escalation *widens* it. If your thesis was "the regime tightens" and the politics turn the other way, the overhang you were short *evaporates*.
- **The buyer restructures around the wire.** Sometimes a deal is saved by carving out the sensitive US unit entirely, or by bringing in a domestic co-investor to dilute foreign control below the "control" threshold. If the structure changes to dodge CFIUS, re-underwrite from scratch.

The meta-lesson ties back to the spine of this whole series: a rule changes the *game*, the market discounts the *expected* outcome of that rule before it fully bites, and your job is to read the rule earlier and size the repricing better than the crowd. CFIUS is one of the purest examples in markets — a single committee's judgment can override every commercial party at the table, and the only edge is to have priced that judgment before the order drops.

## Further reading & cross-links

- [Merger arbitrage: trading regulatory deal risk](/blog/trading/law-and-geopolitics/merger-arbitrage-trading-regulatory-deal-risk) — the spread-as-probability mechanics that CFIUS feeds into.
- [Export controls and the chip war](/blog/trading/law-and-geopolitics/export-controls-and-the-chip-war) — the technology leg of US–China decoupling; outbound screening is its capital twin.
- [Supply-chain geopolitics: reshoring, friendshoring, and critical minerals](/blog/trading/law-and-geopolitics/supply-chain-geopolitics-reshoring-friendshoring-and-critical-minerals) — where national-security logic reshapes whole sectors.
- [How law moves markets: the transmission chain](/blog/trading/law-and-geopolitics/how-law-moves-markets-the-transmission-chain) — the spine: rule → policy → flows → price → the trade.
- [The law, policy, and geopolitics playbook](/blog/trading/law-and-geopolitics/the-law-policy-and-geopolitics-playbook) — the capstone synthesis that pulls every veto point together.
- For the fund-level view of deal-risk and event positioning, see the [hedge-funds](/blog/trading/hedge-funds) and [cross-asset](/blog/trading/cross-asset) series.
