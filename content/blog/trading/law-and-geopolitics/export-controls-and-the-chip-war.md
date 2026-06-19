---
title: "Export Controls and the Chip War: How Export Law Redrew the Semiconductor Map"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "How US export-control law — the Entity List, the Foreign Direct Product Rule, and the October 2022 and 2023 chip controls, plus the CHIPS Act — turned semiconductors into the front line of great-power competition and repriced Nvidia, ASML, SMIC, and TSMC."
tags: ["regulation", "geopolitics", "export-controls", "semiconductors", "chip-war", "entity-list", "fdpr", "chips-act", "nvidia", "asml", "tsmc", "investing"]
category: "trading"
subcategory: "Law & Geopolitics"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — US export-control law turned the most globalized industry on earth, semiconductors, into the front line of great-power competition. A handful of legal instruments — the **Entity List**, the **Foreign Direct Product Rule (FDPR)**, and the **October 7, 2022** chip controls tightened in **October 2023** — plus the **\$52bn CHIPS Act** subsidies, reshaped where advanced chips can be made and sold.
>
> - An export control is a **license requirement** to ship a good or technology abroad. The US Commerce Department's **Bureau of Industry and Security (BIS)** runs it under the **Export Administration Regulations (EAR)**.
> - The FDPR is the long-arm mechanism: it lets US law control a **foreign-made** product if it was built with US tools, software, or design IP. That is why a chip a Taiwanese fab makes for a Chinese customer can still need a US license.
> - The controls cost the **controllers' own firms** real revenue. Nvidia was barred from selling its top AI GPUs to China; ASML faces a China revenue cliff. China is roughly **20-25%** of Nvidia's data-center demand and **25-30%** of ASML's system sales — both now restricted.
> - The one number to remember: **TSMC makes around 90% of the world's leading-edge (sub-7nm) chips.** Control the tools that feed that node and you control the modern AI economy.

On the afternoon of **October 7, 2022**, the US Commerce Department published a set of rules that, on paper, were a dry amendment to an export regulation almost no investor reads. In practice they were one of the most consequential market events of the decade. Overnight, the most advanced artificial-intelligence chips and the machines used to make them became illegal to sell into the world's second-largest economy without a US government license — and the license came with a *presumption of denial*, which is bureaucratic English for "no."

The repricing was immediate and brutal where it bit. Chip-equipment makers — the companies that sell the multimillion-dollar machines that etch circuits onto silicon — sold off. Nvidia, whose graphics processors had quietly become the engine of the AI boom, would within months design a deliberately weakened chip just to keep selling *something* into China, only to watch the government ban that one too a year later. A Chinese national champion, SMIC, suddenly had a hard ceiling on how advanced its chips could get. And a single company on a single island — Taiwan Semiconductor Manufacturing Company, TSMC — found itself even more central to the global economy than it already was, because it was the one place on earth that could reliably make the chips everyone needed.

This post is a case study in how **law moves markets**. It is not about whether the controls are wise or fair — that is a political question, and reasonable people disagree. It is about the *mechanism*: how a license rule, an enforcement list, and one clever piece of legal long-arm jurisdiction repriced four of the most important companies in technology, split a single global supply chain into two, and turned export law into industrial strategy. If you can read this transmission chain, you can read the next chip headline — and size what it is worth.

![Export-control transmission chain from US export law through BIS to named firms and node thresholds to the repricing of Nvidia ASML SMIC and TSMC](/imgs/blogs/export-controls-and-the-chip-war-1.png)

## Foundations: what an export control actually is

Start from zero. An **export control** is a rule that says: before you ship a particular good, piece of software, or technical knowledge across a border, you must get the government's permission. That permission is called an **export license**. Most exports need no license at all — you can ship sneakers or soybeans freely. But a slice of goods is *controlled*: weapons, obviously, but also "dual-use" items — things with both civilian and military uses, like a high-end chip that can run a video game *or* train a missile-guidance model.

Think of it like a bouncer at a door. Most people walk in. But for a specific list of items going to a specific list of destinations or buyers, the bouncer (the government) checks your ID, and can say yes, yes-with-conditions, or no. Export controls are the legal machinery of that bouncer.

In the United States, the bouncer for dual-use items is the **Bureau of Industry and Security**, or **BIS** — an agency inside the Department of Commerce. (Pure weapons are controlled separately by the State Department under a different regime; chips and chipmaking gear are dual-use, so they fall to BIS.) The rulebook BIS administers is the **Export Administration Regulations**, the **EAR**. The EAR is where every item that needs a license is listed, where every country is sorted into a risk tier, and where the penalties for shipping without a license — fines, prison, being cut off from US supply yourself — are written down.

How does the EAR actually decide if your shipment needs a license? It runs on a three-part lookup that is worth understanding, because the chip controls are just a set of edits to it. First, *what is the item?* Every controlled good gets a code called an **Export Control Classification Number (ECCN)** on a master list called the Commerce Control List — there are specific ECCNs for advanced computing chips and for semiconductor manufacturing equipment. Second, *where is it going?* Countries are sorted into groups, and a given ECCN may need a license for some destinations and not others. Third, *who is the buyer, and what is the end use?* Even an otherwise-fine item can require a license if the customer is flagged or the use is a concern (for example, supporting a weapons program or a restricted fab). The chip controls work by tightening all three knobs at once: they edited the ECCN definitions for advanced chips and tools, set China as the controlled destination, and added end-use and end-user restrictions on top.

When a license *is* required, BIS reviews the application against a stated **licensing policy** — and the policy language is where the bite lives. Some categories get "case-by-case" review; the advanced-chip and advanced-tool categories for China carry a **presumption of denial**, meaning the default answer is no and the applicant must overcome it. Investors should read the licensing-policy language, not just the headline, because "license required, case-by-case" and "license required, presumption of denial" are worlds apart for the revenue at stake. A presumption of denial is, for practical purposes, a ban with a paperwork ritual attached.

It helps to see how the *shape* of US controls evolved, because the chip war represents a genuine change in kind. The oldest model was **country-based**: an embargo on an entire nation (Cuba, North Korea). The next was **firm-based**: the Entity List, which targets a named buyer without embargoing its whole country. The newest, and the one the chip controls pioneered at scale, is **capability-based** — controlling an entire *technology threshold* (chips above a certain compute-and-interconnect spec, tools below a certain node) regardless of the buyer's name or the chip's brand. Capability-based controls are far wider than firm-based ones, because they sweep in *every* product that crosses the technical line and *every* buyer in the controlled destination. That is why the October 2022 rules were such a step change: they moved US export policy from "block this company" to "deny this entire capability to this country."

Within the EAR, two instruments matter most for the chip war, and you must understand both.

**The Entity List.** This is a published list of specific named companies, research labs, and organizations that BIS has decided pose a national-security or foreign-policy concern. Once a firm is on the Entity List, US suppliers need a license to ship almost anything to it — and again, the license is usually presumed denied. It is a targeted weapon: instead of banning a *product* or a *country*, you ban a *named buyer*. Huawei, China's telecom and phone giant, is the most famous addition; SMIC and dozens of Chinese AI and supercomputing firms followed.

The Entity List is precise but porous on its own. Precise, because it names the exact firm, so the revenue at risk is bounded by that firm's purchases. Porous, because a listed company can — absent further rules — buy through an unlisted subsidiary, a distributor, or a shell, and because the listing only binds *US suppliers* shipping *US items*. That porousness is exactly the gap the Foreign Direct Product Rule was built to close: when BIS listed Huawei, it found that Huawei could still get chips made for it by foreign foundries using US tools, so it bolted an FDPR on top of the listing to reach those foreign-made chips. The Entity List and the FDPR are therefore not rivals but a *combination* — the list names the target, the FDPR extends the blockade to foreign-made goods bound for that target.

**The Foreign Direct Product Rule, the FDPR.** This is the clever, controversial part, and the source of US export law's astonishing global reach. We will spend a whole section on it below, because it is the hinge of the entire story. In one sentence: the FDPR lets US law control a product **made entirely outside the United States** if that product was produced using US-origin tools, software, or design technology. It converts American dominance of a few upstream layers of the chip stack into jurisdiction over the whole chain.

Now layer on the specific actions. On **October 7, 2022**, BIS used these tools together for the first time at scale against an entire technology category. The rules did several things at once:

- They restricted the export to China of the most advanced **AI accelerator chips** — defined not by name but by *technical thresholds* (a combination of computing performance and the speed at which chips talk to each other). Cross the threshold, you need a license.
- They restricted the **semiconductor manufacturing equipment** needed to make advanced logic chips (below roughly the 14-16 nanometer node), advanced memory, and the like.
- They imposed a striking **"US persons" rule**: American citizens and green-card holders could not support the development or production of certain advanced chips at Chinese fabs without a license — which forced a wave of engineers to choose between their employer and their passport.

A year later, on **October 17, 2023**, BIS *tightened* the rules. It closed the gap that Nvidia had used to design a slightly slower, China-legal chip (more on that below), broadened the equipment controls, and added more firms to the Entity List. The pattern — control, adapt, re-control — is itself a lesson: export controls are a moving line, not a one-time event.

Finally, the carrot to the stick. In **August 2022**, the US enacted the **CHIPS and Science Act**, which authorized roughly **\$52bn** in funding — about **\$39bn** in manufacturing subsidies plus research money — *and* a separate **25% investment tax credit** on the cost of building or equipping a US chip fab. Where the export controls deny China the leading edge, the CHIPS Act tries to *re-shore* leading-edge manufacturing to American soil. The European Union followed with its own **EU Chips Act** (around **€43bn** mobilized), and Japan, South Korea, and Taiwan all stood up their own subsidy programs. The world's governments stopped treating chips as just another import and started treating fabs as strategic assets.

> [!note]
> **A vocabulary anchor.** A *node* (like "5nm" or "7nm") is roughly a generation of chipmaking precision — smaller numbers mean smaller, denser, faster, more power-efficient transistors. *Leading-edge* means the smallest nodes in production (today around 3-5nm); *mature* or *legacy* means older, larger nodes (28nm and up) that still run cars, appliances, and power systems. The chip war is fought almost entirely over the *leading edge*. Mature-node chips are largely uncontrolled — which is why "China can't make chips" is wrong, and "China can't make *the most advanced* chips at scale" is closer to right.

## How the control tools stack up

The instruments are not interchangeable; they form a ladder of escalating reach. The weakest rung is the plain EAR license requirement — ask BIS before you ship a listed item. Above it sits the Entity List, which cuts off a *named* buyer. Above that sit the *node thresholds* of the 2022/2023 rules, which restrict an entire *category* of chip and tool by technical spec rather than by name. And at the top sits the FDPR, which extends all of the above to *foreign-made* goods, giving US law a reach no other tool has.

![Control-tools ladder from EAR license rule to the Entity List to node thresholds to the Foreign Direct Product Rule and license review](/imgs/blogs/export-controls-and-the-chip-war-2.png)

The ladder matters for reading the news. When BIS adds a company to the Entity List, the affected revenue is bounded by that company's purchases. When BIS changes a *node threshold*, the affected revenue is bounded by an entire product line across all Chinese buyers — a much bigger number. And when BIS extends a rule via the FDPR, the affected revenue includes sales by *non-US* companies, which is how a rule written in Washington can dent the earnings of a firm in the Netherlands or Taiwan. As an investor, your first question on any chip-control headline is: *which rung of the ladder is this, and therefore how wide is the revenue at risk?*

## The FDPR: how US law reaches a chip it never touched

Here is the mechanism that makes the whole thing work, and the one most people get wrong. Naively, you might think US export law can only control goods made in the US, or goods with a lot of US content inside them. For decades, that was roughly true — there was a "de minimis" rule that controlled foreign goods only if more than 25% of their value was US-origin. A chip designed in California but fabricated in Taiwan might fall under US rules; a chip designed and built entirely abroad would not.

The Foreign Direct Product Rule rewrote that logic. It says, in effect: *if a foreign-made product is the **direct product** of US-origin technology or software — or is made by a **plant** that is itself the direct product of US technology — then that foreign-made product is subject to US export controls.* The "US content" of the final chip can be zero. What matters is whether US technology sits *anywhere in the toolchain that produced it.*

And US technology sits everywhere in that toolchain. Consider what it takes to make a leading-edge chip:

- **Design software (EDA).** The electronic-design-automation tools that engineers use to lay out billions of transistors are dominated by a handful of firms, of which **Cadence** and **Synopsys** are American. There is essentially no leading-edge chip designed today without US EDA software touching it.
- **Manufacturing tools.** The deposition, etch, and inspection machines on a fab line come heavily from US firms like **Applied Materials**, **Lam Research**, and **KLA**. Even the Dutch lithography giant **ASML** — which makes the machines that print the circuit patterns — relies on US-origin components and software.
- **Design IP.** Reusable circuit blocks and instruction-set architectures often carry US-origin technology.

So when BIS extends a control via the FDPR, it is pulling on a thread that runs through almost every advanced chip on the planet. A fab in Taiwan can be a wholly Taiwanese company making a chip for a wholly Chinese customer — and still need a US license, because the chip is the direct product of US-origin design software and tools. That is the long arm. It is why the rules bind firms that have no factories, no employees, and no listed stock in the United States.

![FDPR mechanism showing US EDA software tools and IP feeding a foreign fab whose chip for a listed customer is blocked while an allowed customer needs no license](/imgs/blogs/export-controls-and-the-chip-war-3.png)

The genius — and the controversy — of the FDPR is leverage. The US controls maybe a fifth of the *value* of the chip stack (design tools, some equipment), but because that fifth is *upstream* and *unsubstitutable in the near term*, controlling it gives effective control over the whole. It is the supply-chain equivalent of controlling a mountain pass: you don't need to own the valley if you own the only road in.

#### Worked example: sizing the FDPR's reach versus de minimis

Suppose a Taiwanese fab makes a custom AI chip for a Chinese cloud company. The chip's bill of materials is \$300 of value: \$30 of US-origin design-IP and software embedded in the design, \$270 of non-US wafers, packaging, and labor. Under the old **de minimis** rule, US content is \$30 / \$300 = **10%**, comfortably below the 25% threshold — *no US license needed.* The chip ships freely.

Now apply the **FDPR**. The question is no longer "what fraction of the chip's *value* is American?" It is "was the chip a *direct product* of US-origin technology?" The fab used US EDA software to design it and US-origin tools to etch it — so **yes**. The 10% value share is irrelevant; the chip is now subject to US controls, and shipping it to a listed customer needs a license that will be denied.

The intuition: de minimis counts *dollars of American content*; the FDPR counts *whether American technology was in the room when the thing was made.* That shift from a value test to a technology-touch test is what gave US export law global reach over chips.

This reach is genuinely contested, and an analyst should hold the tension rather than resolve it. Supporters argue the FDPR is the only tool with enough leverage to deny a capability that is built across a dozen countries — without it, a control on US shipments would simply reroute demand through a foreign fab and accomplish nothing. Critics, including some allied governments and many of the controlled firms, argue that extending US jurisdiction over goods with little or no US content strains the principle that a country regulates its own exports, and risks accelerating exactly the outcome it fears: foreign customers and even allies "designing out" US technology to escape the long arm. That second-order risk — *de-Americanization*, the slow substitution of US tools and IP with non-US alternatives precisely to avoid the FDPR — is the most important long-run uncertainty in the whole space. The controls work today because the US-controlled layers are unsubstitutable in the *near* term; their durability depends on whether that stays true. If China (or a US ally hedging its bets) builds a credible non-US EDA tool or lithography source over a decade, the chokepoint erodes from the inside. The leverage is real, but it is a *depreciating asset* if it provokes its own replacement.

## The Nvidia story: banned, redesigned, banned again

No single company illustrates the controllers'-own-pain problem better than Nvidia. Nvidia designs the graphics processing units (GPUs) that turned out to be ideal for training and running AI models. By 2022-2023, its data-center business — selling AI accelerators to cloud providers and enterprises — had become a money machine, and China was a large slice of it.

The October 2022 rules barred the export to China of Nvidia's top data-center GPUs, the **A100** and the newer **H100**, by setting performance-and-interconnect thresholds those chips exceeded. Nvidia's response was textbook corporate adaptation: it designed the **A800** and **H800** — chips deliberately throttled (mainly on the chip-to-chip interconnect speed) to fall *just under* the 2022 thresholds, so they were legal to sell into China. For about a year, that workaround preserved a chunk of the business.

Then came **October 2023**. BIS rewrote the thresholds specifically to capture the A800 and H800, and added a "performance density" criterion so a chipmaker couldn't simply trim one spec to slip under. The China-legal chips were now China-illegal. Nvidia later designed *further* down-specced parts (the H20 and similar) to chase whatever remained legal, in a cat-and-mouse that continued as the rules evolved. The lesson for investors: when a control is written by *threshold*, the target firm will engineer to the threshold — so the durability of the revenue hit depends on whether the regulator updates the threshold faster than the firm can re-spec. In Nvidia's case, the regulator kept up.

This cat-and-mouse is itself a structural feature worth pricing, not noise. A *threshold-based* control creates a predictable dance: the rule sets a line, the firm engineers a part just under it, the firm books a quarter or two of China revenue on that part, the regulator closes the gap, repeat. Each cycle the China-legal part is *worse* (more throttled) than the last, so even when a workaround exists, the trend is toward thinner products and thinner margins in China. An investor should treat each "Nvidia designs a new China chip" headline not as a reprieve but as one more step down the product ladder — a temporary, depreciating revenue stream, not a restored one. The honest model is a China data-center line that *ratchets down* over successive control rounds, with each workaround buying time rather than reversing the trend.

The **"US persons" rule** deserves its own mention because it shows how deep the controls reach. By restricting US citizens and green-card holders from supporting advanced chip development at Chinese fabs without a license, the October 2022 package effectively forced a wave of senior engineers and executives at Chinese chip firms — many of them US passport holders trained in Silicon Valley — to choose between their employer and their immigration status. Talent, not just tools, became a controlled input. This is a reminder that an export control is not only about physical goods crossing a border; it can reach *services, software, and human expertise*, and those softer controls can be just as binding as a ban on a machine, because a fab without experienced engineers cannot run a leading-edge line no matter what equipment sits on the floor.

#### Worked example: Nvidia's China revenue at risk

Use round, illustrative figures anchored to the public range. Say Nvidia's data-center segment runs at an annualized **\$48bn** of revenue, and management has indicated China is historically around **20-25%** of data-center sales. Take the midpoint, **22%**:

> China data-center revenue ≈ \$48bn × 0.22 = **\$10.6bn / year**

Now, not all of that vanishes — Nvidia can sell *some* down-specced, license-free parts, and can redirect some supply to non-China customers who are demand-constrained. Suppose the controls strand **60%** of the China data-center revenue (the high-end parts) and Nvidia recovers half of the *rest* by reallocating chips to other buyers:

> Stranded high-end: \$10.6bn × 0.60 = **\$6.4bn** at risk
> Recoverable via reallocation: (\$10.6bn − \$6.4bn) × 0.50 = **\$2.1bn** recovered elsewhere
> Net revenue at risk ≈ \$6.4bn − \$2.1bn ≈ **\$4.3bn / year**

On a company with \$48bn of data-center revenue, a ~\$4bn net hit is real but survivable — roughly **9%** of the segment — especially when overall AI demand is growing fast enough to backfill the lost units. The intuition: the *gross* China number looks alarming, but the *net* hit is gross-minus-reallocation, and in a supply-constrained boom a chipmaker can sell every chip it makes to *someone*. That asymmetry is exactly why Nvidia's stock could absorb the controls and keep climbing.

#### Worked example: the margin and ASP effect of losing the China market

Revenue is only half the story; the controls also bite the *mix*, and mix drives margin. Consider a stylized chipmaker that sells two products: a top-tier accelerator at an average selling price (**ASP**) of **\$25,000** with a **70%** gross margin, and a down-specced, China-legal accelerator at an ASP of **\$12,000** with a **55%** gross margin (lower price, lower margin, because the throttled part competes harder and carries the same fixed costs spread over a cheaper unit).

Before the controls, suppose China buys **100,000** top-tier units a year:

> China gross profit before = 100,000 × \$25,000 × 0.70 = **\$1.75bn**

After the controls, the top-tier part is banned in China. The firm pivots those Chinese customers to the down-specced part — but a worse product sells fewer units (some buyers defer, some go to a domestic substitute), say **60,000** units:

> China gross profit after = 60,000 × \$12,000 × 0.55 = **\$0.40bn**

> Lost gross profit ≈ \$1.75bn − \$0.40bn = **\$1.35bn** — a **77%** collapse in China gross profit, far worse than the revenue decline alone.

Notice the leverage: revenue fell from \$2.5bn to \$0.72bn (−71%), but gross *profit* fell 77%, because the firm lost its *richest* product in that market and replaced it with a thinner one. The intuition: export controls that force a chipmaker down the product ladder hit margins harder than revenue, because the controlled high-end is almost always the high-margin end — losing the China market is really losing the *China mix*. A practitioner models the controls through the income statement, not just the top line.

## ASML: the lithography chokepoint and a China revenue cliff

If Nvidia is the demand-side story, ASML is the supply-side one. ASML, a Dutch company, makes the **lithography** machines that print circuit patterns onto silicon wafers. It has a *monopoly* on the most advanced type — **extreme ultraviolet (EUV)** lithography — the only technology capable of mass-producing the smallest, most advanced nodes. No EUV, no leading edge. ASML also makes the older **deep ultraviolet (DUV)** machines, which can make somewhat-advanced chips with clever multi-step "multi-patterning" tricks, and which China can use to push toward 7nm-class chips.

Here the controls reach ASML through *allied alignment*, not just the FDPR. Because ASML is Dutch, the cleanest way to control its exports is for the **Netherlands** to impose its own rules — which, after extended diplomacy, it did, restricting first EUV (never sold to China) and then a tranche of advanced **DUV** machines. **Japan**, home to other key tool and material makers (Tokyo Electron, plus the photoresist and specialty-chemical firms that feed every fab), aligned its own controls in 2023. The result is a coordinated, multi-government regime: the US writes the template, allies adopt parallel rules, and the FDPR backstops anything that slips through. The chip war is a *coalition*, not a US solo act — and that coalition is the reason the controls bind.

Why did the allies go along, given that their own champions (ASML, Tokyo Electron) bear the revenue cost? Two reasons that matter for forecasting the regime's stability. First, the FDPR makes non-cooperation partly pointless: even if the Netherlands declined to restrict ASML, the FDPR could still capture ASML machines that contain US-origin components, so the practical choice for The Hague was "set our own terms" versus "have Washington set them for us." Second, the allies share the underlying security concern, even when they quarrel over scope and timing. The friction shows up in the *pace* — allied governments tend to move slower and carve more exemptions than the US would like — which is why the *speed* of allied alignment is a tradeable variable: a faster-than-expected Dutch or Japanese tightening is a negative surprise for ASML and Tokyo Electron, a delay or a carve-out is a relief rally. The coalition is real but not frictionless, and the seams are where the surprises live.

For ASML, the market mechanism is a **China revenue cliff**. China had been buying older DUV machines heavily — partly stockpiling ahead of restrictions — so for a stretch China was an outsized share of ASML's system sales. As the DUV controls phased in, that demand was scheduled to fall off a cliff. The stock reprices not on the day the rule is signed but as the *forward* China revenue line is marked down in analysts' models — the classic "expectations drift" of any rule-driven repricing.

![Global foundry revenue share showing TSMC at 64 percent versus Samsung SMIC UMC GlobalFoundries and others](/imgs/blogs/export-controls-and-the-chip-war-6.png)

That concentration is the backdrop against which every chip control is written, and it explains the structure of the whole map. When one company holds roughly 64% of all foundry revenue and the overwhelming majority of the leading edge, a control on the *tools* that company depends on becomes a control on the *entire* advanced-chip economy — there is no fragmented field of fabs to route around. The remaining share is split among Samsung (the only other firm with credible leading-edge ambitions), SMIC and UMC (mature and mid-node), GlobalFoundries (deliberately exited the leading-edge race to specialize), and a long tail. A reader should resist the temptation to treat "foundry" as a commodity industry: it is a steep pyramid, and the controls are aimed squarely at the apex.

#### Worked example: ASML's China exposure and the DUV cliff

Say ASML books **\$30bn** of system sales in a peak year, and in that year China is an unusually high **29%** of system revenue (a stockpiling year):

> China system revenue ≈ \$30bn × 0.29 = **\$8.7bn**

Now the DUV controls phase in. Suppose the restricted machines are **40%** of that China figure, and the rest (mature-node and serviceable older tools) stays legal:

> Restricted China sales = \$8.7bn × 0.40 = **\$3.5bn**

If China's share then normalizes from 29% back toward a structural **15%** as the stockpiling unwinds *and* the restricted tranche disappears, ASML's China revenue could fall from \$8.7bn toward roughly **\$4.0-4.5bn** — a **~50%** China revenue decline over a couple of years. The offset: ASML has a multi-year **order backlog** for EUV from non-China customers (TSMC, Samsung, Intel, all racing to build leading-edge capacity), and *service* revenue on its huge installed base is sticky. The intuition: the China cliff is a real headwind, but ASML's monopoly on the *tool everyone else needs* means the controls partly *redirect* demand rather than destroy it — the leading-edge buildout elsewhere backfills the China hole.

There is a subtlety in ASML's case that rewards attention: the *type* of restriction matters more than its existence. EUV was never sold to China, so EUV controls cost ASML *nothing it had* — they merely formalized a status quo. The painful part is the **DUV** tranche, because China *was* a heavy DUV buyer. So when a headline says "the Netherlands restricts ASML exports to China," the tradeable question is *which tool* — an EUV-only restriction is largely symbolic for ASML's earnings, while a DUV restriction marks down a real revenue line. This is a general lesson for reading chip-tool controls: a control on a tool the target *wasn't getting anyway* (EUV) is cosmetic; a control on a tool the target *was buying* (DUV) is the one that moves estimates. Always ask what the control actually removes from the order book.

A second subtlety: ASML's revenue has a long *recognition lag*. Lithography systems are ordered years before they ship and are recognized as revenue on delivery, so a control announced today shows up in reported sales only after the backlog ahead of it clears. That lag is why the stock reprices on the *guidance* — when management cuts the forward China revenue assumption — rather than on the rule's signing date. The market is discounting a revenue line that won't actually fall for several quarters.

## SMIC: a capped national champion

On the receiving end sits **SMIC** — Semiconductor Manufacturing International Corporation, China's largest foundry, added to the Entity List in 2020. The controls deny SMIC the EUV machines it would need to mass-produce the smallest nodes, and restrict the advanced DUV and the deposition/etch tools beyond a point. SMIC stunned observers by producing a **7nm-class** chip using DUV multi-patterning — proving that *node progress is not impossible without EUV*. But there is a catch the headlines often miss: making 7nm on DUV is **expensive and low-yield**. Each chip requires more passes through the machines, more masks, more time, and a higher scrap rate. You can do it; you cannot do it *cheaply, at the volumes, or at the next node down* without the controlled tools.

So SMIC's story is the nuanced middle of the "China can't catch up" debate. China *can* climb — slowly, expensively, one painful node at a time — but the controls impose a **cost and volume tax** that widens, not closes, the gap with TSMC at the bleeding edge. For an investor, SMIC is a *volume-up, node-capped* play: domestic demand surges as Chinese buyers are forced onto local supply, but margins and the technology ceiling are pinned by the missing tools.

It is worth being concrete about *why* the missing EUV machine matters so much, because it is the physical heart of the control. Lithography prints a circuit pattern by shining light through a mask onto the wafer; the smaller the wavelength of the light, the finer the pattern you can print. EUV uses an extremely short wavelength (13.5 nanometers) that lets a fab print a tiny feature in a *single* exposure. DUV uses a longer wavelength, so to print the same fine feature you must expose the wafer *multiple times* with slightly shifted masks — "multi-patterning." Each extra pass adds cost, adds time, multiplies the number of expensive masks, and multiplies the chances of a defect, which crushes **yield** (the fraction of chips on a wafer that work). So SMIC's DUV-only 7nm is not a free lunch: industry estimates put its yields and costs far worse than TSMC's EUV-based equivalent. The control doesn't make the next node *impossible* — it makes it *uneconomic at scale*, which for a commercial foundry competing on cost is nearly the same thing. That distinction — impossible versus uneconomic — is the single most misunderstood point in the whole chip-war discourse, and getting it right is the difference between a good and a bad China-tech thesis.

## TSMC and the silicon shield

And then there is TSMC. The single most important fact in the entire chip war is one number: **TSMC makes around 90% of the world's leading-edge (sub-7nm) chips**, and roughly **64%** of *all* foundry revenue. When Apple, Nvidia, AMD, and the rest need their most advanced chips made, they go to TSMC — there is, for the leading edge, no real second source at scale.

![Share of leading-edge sub-7nm foundry capacity showing TSMC at about 90 percent versus everyone else at about 10 percent](/imgs/blogs/export-controls-and-the-chip-war-7.png)

This concentration is the source of two ideas you will hear constantly. The first is the **"silicon shield"** — the theory that Taiwan's centrality to the global chip supply makes it too economically important for any actor to risk a conflict that would shatter TSMC's fabs, because the resulting chip famine would crater the world economy. The second is the corresponding **risk premium**: because so much of modern technology runs through a small island in a geopolitically tense strait, the entire chip complex carries a *Taiwan tail risk* that flares on any escalation headline. The same concentration is both a stabilizer (everyone needs the fabs intact) and a vulnerability (everyone is exposed if they aren't).

For TSMC itself, the export controls are *mostly a tailwind with a compliance cost.* TSMC's direct China-customer revenue is a modest slice; the bulk of its leading-edge work serves US and allied design houses. The controls force TSMC to *screen its customers* — it must refuse orders from Entity-List firms and from designs that would breach the rules — which is an administrative burden but not a revenue threat. Meanwhile the controls *entrench* TSMC's position as the indispensable, trusted supplier to the US-aligned world, and CHIPS Act money helps fund its new fabs in Arizona. TSMC earns a **safe-supplier premium**: in a bifurcating world, being the fab everyone trusts is worth more, not less.

The screening obligation is more than paperwork, though, and it has teeth. After the Huawei episode — where a TSMC-made chip ended up in a Huawei device despite the rules — BIS made clear that foundries are expected to *know their customer* and trace where their advanced chips end up. A foundry that ships a leading-edge chip to a shell company that turns out to front for a listed firm risks its own access to US tools. So TSMC and Samsung now run real compliance machinery: customer vetting, end-use certifications, and a willingness to refuse profitable orders rather than risk the EAR. This is a hidden second-order effect of the controls — they conscript the *foundries themselves* into the enforcement apparatus, because the foundries' own survival depends on US tools they cannot replace. The chokepoint enforces itself: ASML and the US toolmakers don't need to police every chip, because the fabs that depend on their machines police it for them.

The CHIPS Act dimension deserves a clear-eyed note too. Re-shoring leading-edge fabs to the US is expensive and slow — Arizona fabs have faced higher costs and labor and supply-chain frictions relative to Taiwan, and the leading edge of any fab company tends to stay closest to its home R&D. The realistic outcome is *diversification*, not relocation: the US-aligned world ends up with *some* domestic leading-edge capacity as insurance, while the center of gravity stays in Taiwan. For an investor, that means the "TSMC is irreplaceable" thesis and the "CHIPS Act re-shoring" thesis are not contradictory — the subsidies buy redundancy at the margin without dethroning the incumbent. The two figures to watch are TSMC's *leading-edge share* (still ~90%, the moat) and the *trajectory* of US-built advanced capacity (rising from near-zero, the insurance).

![Geopolitical Risk index over time showing elevated readings as the chip controls land relative to the long-run mean of 100](/imgs/blogs/export-controls-and-the-chip-war-8.png)

The controls did not happen in a vacuum. They landed in a decade of structurally elevated **geopolitical risk** — the index of geopolitical tension has sat well above its long-run average through the 2020s, spiking on each new shock. Chip controls are one expression of that backdrop: when great powers compete, they reach for the economic chokepoints they control, and the US reached for the one it had — the upstream layers of the semiconductor stack.

There is a deeper strategic logic worth naming, sometimes called the **"small yard, high fence"** approach: rather than decoupling the whole \$600bn-plus chip trade, the US defines a *narrow* set of capabilities (leading-edge AI compute and the tools to make it — the small yard) and walls those off as completely as possible (the high fence), while leaving the vast mature-node trade largely open. The strategy is an answer to the controllers'-own-pain problem: confine the revenue sacrifice to the narrowest slice that still denies the critical capability. Whether the yard stays small is itself a market variable — each tightening (the 2023 round, subsequent updates) expands the yard, and each expansion widens the revenue at risk. An investor tracks the *width of the yard* as a leading indicator of how much chip revenue is in the crosshairs.

## The bifurcation: one chain becomes two

Step back and the cumulative effect of all this is **structural**: a single, hyper-globalized supply chain is splitting into two partially separate ones. For decades, chips were the showcase of globalization — designed in California, patterned with Dutch machines, fabricated in Taiwan, packaged in Malaysia, assembled into devices in China, sold everywhere. The controls drive a wedge through that chain.

![One integrated global chip supply chain on the left splitting into a US-aligned leading-edge bloc and a China-aligned domestic bloc on the right](/imgs/blogs/export-controls-and-the-chip-war-4.png)

On one side, a **US-aligned bloc** gets the leading edge: EUV lithography, the most advanced AI GPUs, the best EDA tools — available to the US, its allies, and trusted firms, ring-fenced from China. On the other, a **China-aligned bloc** is pushed toward **domestic substitution**: SMIC and the homegrown tool maker SMEE racing to replicate capabilities, capped at mature and near-7nm nodes, subsidized heavily by Beijing, and selling primarily into China's own enormous market. The two blocs still trade in the *uncontrolled* mature-node chips that run cars and appliances — full decoupling is neither the goal nor realistic — but at the *leading edge*, two parallel ecosystems are forming.

This is the deepest market consequence, because it changes *structure*, not just one quarter's revenue. A bifurcated chain means duplicated capacity (two sets of fabs where one would do), higher costs (lost economies of scale), and a durable *regulatory risk premium* baked into every chip name with cross-bloc exposure. It also creates winners: the subsidy beneficiaries building re-shored capacity, the equipment makers selling into *both* blocs' buildouts, and the trusted fabs that anchor the US-aligned side.

The bifurcation is reinforced by a **subsidy race** that is itself a market force. The US CHIPS Act (~\$52bn) was the starting gun; the EU Chips Act (~€43bn mobilized), Japan's multi-trillion-yen support for re-shoring and for a new domestic leading-edge venture, South Korea's tax incentives, and China's enormous state-backed investment funds all followed. The economic effect of every government subsidizing fabs at once is *over-investment* relative to a single global market — which is great for the equipment makers who sell the tools regardless of who builds, and a longer-term risk of *mature-node oversupply* as China in particular floods the legacy-chip market with subsidized capacity. So the subsidy race has two market signatures: a near-term tailwind for tool vendors, and a watch-item for a future glut in the uncontrolled mature nodes where China is building hardest. The same policy that denies China the leading edge funnels its capital into the trailing edge — and that displaced capital could pressure prices in the very chips the controls *don't* touch.

#### Worked example: the CHIPS Act subsidy and tax credit on a \$20bn fab

Here is the carrot in dollars. Suppose a chipmaker builds a leading-edge fab in the US for a total project cost of **\$20bn**. The CHIPS Act offers two stacking incentives:

> **1. Direct manufacturing grant.** CHIPS grants have run roughly **5-15%** of project cost (the program is oversubscribed, so awards vary). Take **10%**:
> Grant ≈ \$20bn × 0.10 = **\$2.0bn**
>
> **2. Investment tax credit (ITC).** A **25%** credit applies to the *qualified investment* — the equipment and building, say **\$16bn** of the \$20bn:
> ITC ≈ \$16bn × 0.25 = **\$4.0bn**

> **Total public offset ≈ \$2.0bn + \$4.0bn = \$6.0bn**, or about **30%** of the \$20bn build cost.

That is enormous. A 30% reduction in the up-front cost of a fab is the difference between a US project that pencils out and one that doesn't — which is precisely the point of the subsidy. The intuition: export controls *push* leading-edge capacity away from China; the CHIPS Act *pulls* it toward the US. Together they don't just block — they relocate the map.

![Revenue-at-risk matrix for Nvidia ASML SMIC and TSMC showing China exposure what the rule hits and the net direction for each](/imgs/blogs/export-controls-and-the-chip-war-5.png)

## Common misconceptions

**"Export controls only hurt the target."** This is the most expensive misconception in the space. Controls cost the *controllers' own firms* real revenue, because the target is also a *customer*. Nvidia faced roughly \$10bn of gross China data-center revenue at risk; ASML faces a China revenue cliff from what had been ~29% of system sales in a peak year; the broader US chip-equipment industry lost billions in Chinese orders. The controls are a deliberate trade — accept revenue pain at home to deny capability abroad — and the home-side pain is a feature, not a bug, of the policy. An investor who models only the target's loss and not the controller's revenue hit will misprice both sides.

**"China can't catch up."** Too strong. SMIC produced a 7nm-class chip on DUV multi-patterning, proving node progress is possible without EUV. The accurate statement is narrower: China can climb the node ladder, but the controls impose a **cost-and-yield tax** that makes each advance expensive and low-volume, and they deny the tools needed for the *next* node down at scale. The gap at the bleeding edge widens even as China makes real mature- and mid-node progress. "Can't catch up" is wrong; "can't catch up *cheaply, at volume, at the leading edge, on this timeline*" is right — and that nuance is the whole investment thesis.

**"The controls are easily evaded."** Smuggling and transshipment exist — chips are small and valuable, and some leak through third countries. But the controls bite where it matters most: you cannot smuggle a **fab**. Leading-edge manufacturing requires EUV machines that weigh hundreds of tons, ship in dozens of crates, need constant servicing by the maker's engineers, and number only a handful per year — there is no black market for an ASML EUV system. Evasion can move *finished chips* at the margin; it cannot move the *capability to make them at scale*. That is why the equipment controls, not the chip controls, are the structural ones.

**"It's mainly about today's military."** The controls are aimed less at any current weapon than at the *AI computing base* — the ability to train the largest models, which underpins both economic and military advantage for a decade. That is why the thresholds target *training-class* accelerators and the *tools to make them*, not specific munitions. Reading the controls as ordinary arms control understates their scope: they are an attempt to shape who gets to lead in artificial intelligence.

**"The controls are pure cost to US firms, so they'll be reversed."** This conflates two different things — the *firm's* interest and the *policy's* interest. Yes, the controlled firms lobby against tightening, because the China revenue is theirs to lose. But the controls are a *bipartisan* national-security policy with durable political support, and the firms' lost China revenue has been substantially backfilled by the AI boom, which blunts the lobbying pressure. A reversal would require a genuine diplomatic thaw, not just corporate complaint. The right base case is *sticky controls with episodic tightening*, not imminent rollback — which is why the China-exposed revenue lines are best modeled as structurally impaired, with any thaw as upside optionality rather than the expected path.

## How it shows up in real markets

**The control-announcement move.** Chip-equipment and AI-chip stocks reprice sharply on control headlines, and — crucially — on the *rumors* and *proposals* that precede the final rule. By the time the formal BIS rule is published, much of the move is already in the price, because reporting of a forthcoming tightening leaks for weeks. The tradeable event is often the *leak*, not the *rule*. A practitioner watches BIS rumor flow, allied-government statements (a Dutch or Japanese alignment headline moves ASML and Tokyo Electron), and the quarterly guidance where companies first quantify the China hit.

#### Worked example: turning a control headline into an implied revenue hit

You can back out what the market *thinks* a control is worth using a simple event-study logic. Suppose a chip-equipment maker has a market capitalization of **\$200bn** and, on the day a tightening of China tool controls is reported, its stock falls **6%** while the broad semiconductor index falls **1.5%** (the market-wide move). The **abnormal return** — the part attributable to the news, not the market — is:

> Abnormal return ≈ −6% − (−1.5%) = **−4.5%**
> Value destroyed by the news ≈ \$200bn × 0.045 = **\$9bn**

Now sanity-check that against fundamentals. If the firm trades at roughly **6× sales** (a rich multiple typical of the group), the market is implying a permanent revenue loss of:

> Implied lost revenue ≈ \$9bn ÷ 6 = **\$1.5bn / year**

So the −4.5% move is the market's estimate that the control strips about \$1.5bn of *annual* revenue, capitalized at the firm's multiple. The trade is then a *disagreement* trade: if your bottom-up read of the China order book says the real annual hit is only \$0.8bn, the market over-reacted and you fade the move; if you think it is \$2.5bn, the market under-reacted and you press it. The intuition: a control-day move is a *market-implied revenue estimate in disguise* — divide the abnormal value change by the multiple and compare it to your own number.

**The China-revenue repricing.** The durable repricing is not the one-day gap; it is the slow markdown of the *forward China revenue line* in analysts' models as the controls phase in. ASML's and Nvidia's China exposure got re-rated over *quarters*, as each earnings call updated the share of revenue now restricted. This is "expectations drift" in action: the stock grinds to a new level as the consensus forward estimate migrates, not in a single jump.

**The subsidy beneficiaries.** The CHIPS Act created a cohort of beneficiaries — the firms building re-shored fabs (TSMC Arizona, Intel, Samsung Texas, Micron) and the equipment makers who sell into *every* fab buildout regardless of bloc. The equipment makers are a particularly clean expression: a bifurcated world means *more* fabs get built (duplicated capacity), and every fab — US-aligned or China-aligned — needs deposition, etch, and inspection tools. Some of those tools are controlled into China; many are not. The buildout is a structural tailwind for the toolmakers even as specific China lines get capped.

**The safe-supplier premium and the Taiwan tail.** TSMC trades with a structural tension: a *safe-supplier premium* for being the indispensable, trusted fab, set against a *Taiwan tail-risk discount* that flares on any cross-strait escalation. The two pull in opposite directions, and the net is a name whose multiple is unusually sensitive to geopolitical headlines relative to its rock-solid operating performance.

## The playbook: how to read and trade the chip controls

Pull it together into a repeatable process. Every chip-control headline runs through the same four questions.

**1. Map the control to revenue-at-risk by company.** Identify which *rung of the ladder* the action is (Entity List = one named buyer; node threshold = a whole product category; FDPR extension = non-US firms too) and therefore how wide the revenue at risk is. Then, for each affected firm, compute *gross* China revenue at risk, subtract *reallocation* (chips sold elsewhere in a supply-constrained market) and *license-eligible* sales, to get the *net* hit. The matrix above is the template: who loses China demand (Nvidia, ASML), who is capped (SMIC), who gains a screening cost but a premium (TSMC).

**2. Identify the subsidy beneficiaries.** Run the CHIPS/EU-Chips-Act offset math (a ~30% build-cost reduction on a leading-edge fab is realistic when grant and ITC stack). The beneficiaries are the re-shoring fabs and — more cleanly — the equipment makers who sell into a world that is now building *more* fabs, not fewer.

**3. Express the bifurcation trade.** The structural call is that one chain becomes two: long the *trusted leading-edge* (the indispensable fab, the monopoly toolmaker, the AI-chip leader whose demand outruns the China loss) and long the *equipment buildout* that both blocs feed. Treat *China-exposed revenue lines* as a discount factor, not a thesis-killer, because reallocation and the broader AI boom backfill much of the loss.

**4. Know what invalidates the view.** The bifurcation/safe-supplier thesis weakens if: (a) the controls are *rolled back* in a diplomatic thaw (reversible policy — a deal can lift a rule), repricing the China-exposed names *up*; (b) China achieves *cheap, high-volume* leading-edge production without controlled tools (the cost-and-yield tax breaks down), collapsing the moat; (c) an AI demand *bust* removes the supply constraint, so the China revenue loss can no longer be reallocated and the *gross* hit becomes the *net* hit; or (d) a cross-strait shock turns the Taiwan tail-risk from a discount into a realized loss. Each of these has a watch-item — a thaw shows up in BIS rule revisions and license-grant rates; a China breakthrough shows up in SMIC yield and node disclosures; a demand bust shows up in hyperscaler capex guidance.

A compact way to hold the whole map: line the four names up by *what the rule does to them*. Nvidia and ASML are the **revenue-at-risk** names — the controllers'-own-pain side, where China demand is restricted and the question is how much of the gross loss reallocation can backfill. SMIC is the **capped-champion** name — volume up on forced domestic substitution, but margins and node ceiling pinned by missing tools. TSMC is the **safe-supplier** name — a compliance cost and a Taiwan tail-risk, set against a deepening moat as the indispensable trusted fab. And underneath all four sits the **equipment complex** — the toolmakers who sell into every fab in both blocs, the cleanest expression of a world that is now building more capacity, not less. Map any new headline onto those four buckets and you already know the direction of the trade before you size it.

The deeper lesson, and the reason this is a *case study* rather than a one-off: export law is industrial strategy now. A license rule, an enforcement list, and one piece of long-arm jurisdiction repriced four of the most important companies in technology and bent the global chip map. The mechanism — *law changes the rules of the game; markets discount the change; the practitioner reads it early and sizes it* — is the same mechanism that runs through tariffs, sanctions, and every other rule in this series. Chips are simply the clearest, highest-stakes example of it on the board today. Learn to read this one cleanly and the next chip headline stops being a scary surprise and becomes what it actually is: a rule-change with a knowable revenue consequence, waiting to be sized.

## Further reading and cross-links

- [How Law Moves Markets: The Transmission Chain From Statute to Stock Price](/blog/trading/law-and-geopolitics/how-law-moves-markets-the-transmission-chain) — the spine post; the law-to-policy-to-macro-to-price-to-trade chain this case study runs along.
- [The Legal Architecture of Global Trade](/blog/trading/law-and-geopolitics/the-legal-architecture-of-global-trade) — the WTO, customs, and trade-law backdrop against which export controls are an exception.
- [Taiwan, Semiconductors, and the Most Important Supply Chain on Earth](/blog/trading/law-and-geopolitics/taiwan-semiconductors-and-the-most-important-supply-chain-on-earth) — the geopolitics of TSMC's centrality and the silicon shield.
- [Sanctions Law: OFAC, the SDN List, and Secondary Sanctions](/blog/trading/law-and-geopolitics/sanctions-law-ofac-the-sdn-list-and-secondary-sanctions) — the sister chokepoint tool, with the same long-arm logic applied to finance.
- [The Law, Policy, and Geopolitics Playbook](/blog/trading/law-and-geopolitics/the-law-policy-and-geopolitics-playbook) — the series capstone tying the rule-reading process together.
- [Economic Moats: Durable Competitive Advantage](/blog/trading/equity-research/economic-moats-durable-competitive-advantage) — why a chokepoint like EUV lithography or leading-edge fabrication is the deepest kind of moat.
- [Correlation by Regime: Growth and Inflation](/blog/trading/cross-asset/correlation-by-regime-growth-and-inflation) — how a regulatory regime shift (here, bifurcation) sets the backdrop that cross-asset correlations price against.
