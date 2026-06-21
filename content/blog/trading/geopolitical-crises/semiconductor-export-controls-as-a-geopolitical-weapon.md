---
title: "Semiconductor Export Controls: The Chip War as Geopolitical Strategy"
description: "October 2022: the US effectively banned Nvidia's best chips from China with a single BIS regulation. It was the most significant act of technology warfare since the Cold War. Understanding the chip war tells you where tech geopolitics is going — and which companies win."
date: "2026-06-21"
publishDate: "2026-06-21"
tags: ["geopolitics", "trade-war", "semiconductor", "china", "export-controls", "technology", "chips", "nvidia", "supply-chain"]
subcategory: "Geopolitical Crises"
category: "trading"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — The US weaponized semiconductor export controls in October 2022, effectively cutting China off from the most advanced AI chips and making the chip supply chain the central front of the new cold war between the two largest economies.
>
> - A single BIS regulation on October 7, 2022 blocked Nvidia's A100 and H100 chips from China — triggering a chain of escalations that reshaped the entire global semiconductor industry
> - Nvidia lost roughly \$8.6 billion in annualized China revenue but gained far more from the Western AI boom: stock up 239% in 2023
> - China is spending \$150 billion in state funds to build domestic chip capability, but remains 5-10 years behind TSMC at the leading edge
> - The one fact to remember: the US controls global chip production not through owning the fabs, but through software and equipment that no country can replicate on its own timeline


On October 7, 2022, a Friday morning, the US Bureau of Industry and Security published a 139-page rule in the Federal Register that changed the global technology order. The rule banned Nvidia's A100 and H100 chips — the engines powering the global AI revolution — from being sold to China or Russia. It extended a doctrine called the Foreign Direct Product Rule, which means any chip made anywhere in the world using US software or equipment is now subject to US export controls. It gave TSMC, Samsung, and every other major foundry a legal ultimatum: cut off China or lose access to US technology.

Markets barely noticed that Friday. Nvidia's stock dropped about 12% in the following days, which analysts dismissed as a manageable China exposure. Within twelve months, Nvidia's stock had risen 239%. The "manageable" chip ban had just produced the greatest scarcity premium in technology history.

This post is about how semiconductor export controls became the most powerful geopolitical weapon of the 2020s, why the US holds a nearly unbreakable chokepoint over global chip production, and what all of it means for investors who want to read the chip war as a market signal rather than just a news headline.

![The chip war supply chain from EDA software to AI server with US control points marked](/imgs/blogs/semiconductor-export-controls-as-a-geopolitical-weapon-1.png)

## Foundations: Understanding Chips, Controls, and the Tech Supply Chain

Before diving into the politics, you need to understand why semiconductors are so hard to control — and why control is so effective once you have it.

A semiconductor chip is not one product made in one place. It is the output of a global supply chain where no single country controls every step, but the US and its allies control the most critical ones. Here is how the chain works.

**Step 1: EDA software (Electronic Design Automation):** Before a chip can be fabricated, engineers must design it using highly specialized software. The two dominant vendors are Cadence Design Systems and Synopsys, both American. Without their tools, you cannot design a chip that can be manufactured at a modern foundry. There is no credible non-US alternative. The barrier to replicating this software is not financial — it is decades of accumulated engineering knowledge embedded in the codebase.

**Step 2: Chip architecture:** The logical blueprint for how a chip's circuits are organized — its instruction set architecture — is typically licensed from ARM (a British company now majority-controlled by US-listed SoftBank and deep in the US IP ecosystem) or developed in-house as x86 (Intel/AMD, American). Most AI chips use ARM IP or custom designs licensed under agreements that trigger export controls.

**Step 3: Lithography machines:** To print a chip onto silicon, you need a machine called a lithography tool. The most advanced version — Extreme Ultraviolet (EUV) lithography, which is required for chips below 7 nanometers — is made exclusively by ASML, a Dutch company in Veldhoven, Netherlands. There is no second source. ASML's EUV machines cost around \$350 million each, take years to build, and require thousands of proprietary components. When the US pressured the Netherlands to deny ASML an export license to ship EUV machines to China in 2019, they effectively put a hard ceiling on how advanced Chinese chips could become.

**Step 4: Advanced foundry:** Chip design companies like Nvidia and Apple do not make their own chips. They send designs to specialized manufacturers called foundries. The world's most advanced foundry is Taiwan Semiconductor Manufacturing Company (TSMC), which manufactures more than 90% of the world's most advanced chips. Samsung (South Korea) is a distant second. China's domestic foundry, SMIC (Semiconductor Manufacturing International Corporation), is at least two full generations behind.

**Step 5: Specialized materials:** The chemicals, gases, and raw materials required for chip fabrication — ultra-pure silicon, photoresists, etchants — are dominated by Japanese companies like Shin-Etsu Chemical and ULVAC. Japan joined the US-Netherlands export control coalition in 2023.

This is why the US holds a veto over global chip production without owning a single leading-edge fab. The US controls EDA software, chip architecture licensing, the equipment ecosystem that ASML depends on for its components, and the design firms whose products drive demand. Cut off any one of these, and the chain fails.

Chips are China's largest import category — roughly \$400 billion per year. For context, that is more than China spends on crude oil. When the US targeted this supply chain, it was weaponizing China's largest single point of import dependency.

**Why nanometers matter so much:** When you hear that TSMC is making "3nm chips" and SMIC is stuck at "7nm," it sounds like a small numeric difference. But the physical and commercial gap is enormous. In semiconductor manufacturing, a "node" refers to the size of the smallest features that can be printed on a chip — and smaller features mean you can fit more transistors in the same space.

More transistors per unit area means: more computing power per chip, better energy efficiency (critical for AI data centers running millions of chips), and lower cost per computation. Going from 7nm to 3nm roughly doubles transistor density. At the frontier of AI training — where models like GPT-4 and Gemini Ultra require quintillions of floating-point operations — having chips at 3nm versus 7nm is not a marginal advantage. It is the difference between running a frontier model at all versus not having the compute budget for it.

China being locked at or below 7nm means that for the foreseeable future, Chinese AI companies training frontier models will be doing so with far less compute efficiency than their Western counterparts. This is why the US export control strategy targets process node capability so specifically.

**The interconnect dimension:** Beyond raw chip manufacturing capability, the export controls also restricted high-bandwidth interconnects — specifically the NVLink technology that allows multiple Nvidia GPUs to communicate with each other at very high speed. When you train a large AI model, you need hundreds or thousands of GPUs working together in a coordinated cluster. The interconnect speed between those GPUs determines how efficiently the cluster can function. The A800 and H800 chips Nvidia designed for China compliance had reduced NVLink bandwidth, which meant that even if China could buy these chips, assembling them into effective large-scale AI training clusters was technically more challenging. This is a more subtle but important dimension of the controls' effectiveness.

## The Political Moves: Why the US Weaponized Export Controls

The October 2022 BIS rule did not come out of nowhere. It was the culmination of a strategic realization inside the US national security establishment: China was using access to Western semiconductor technology to build military AI systems, surveillance infrastructure, and high-performance computing capabilities that directly threatened US interests.

**The strategic calculus:** For decades, the US policy toward China on technology was broadly characterized as "engagement" — the belief that economic interdependence and technology access would gradually liberalize China's political system and create a stable, mutually beneficial relationship. By the early 2020s, that framework had collapsed inside the US policy community. China's military modernization, its surveillance state built on AI and big data, its actions in the South China Sea, its treatment of Xinjiang's Uyghur population (where AI-powered surveillance played a documented role), and its "military-civil fusion" doctrine (which explicitly states that Chinese civilian technology companies must support military objectives) all contributed to a view that US technology was being systematically redirected to purposes that threatened US interests.

The specific inflection point was a 2021-2022 US government assessment that Chinese supercomputing centers — many of which were running clusters of Nvidia A100 chips legally purchased through commercial channels — were being used to simulate nuclear weapons, model hypersonic missile trajectories, and train AI systems for autonomous weapons. The A100 was the chip of record for this work. The export control action was designed to cut off that specific capability while maintaining broader commercial technology trade.

**The coordination with allies:** One of the most significant aspects of the chip war is that it is not just a US policy — it is a coordinated multilateral regime. The US worked closely with the Netherlands (ASML's home) and Japan (which supplies critical wafer materials and older chip equipment through companies like Tokyo Electron) to build what amounts to an allied semiconductor technology control bloc. The fact that the Netherlands agreed to restrict ASML's China exports, despite significant commercial cost to a domestic company, demonstrates the seriousness with which US allies view the technology security threat from China.

Japan joined the coalition in 2023, restricting 23 categories of semiconductor equipment exports to China. This matters because Japanese companies like Tokyo Electron, Shin-Etsu Chemical, and Sumco Corporation supply critical materials and equipment that China's domestic chip industry depends on. The coordinated multilateral approach means China cannot simply "shop around" among US allies for what the US won't sell directly.

The specific trigger was a set of intelligence assessments — some public, some not — that China's People's Liberation Army was building AI-enabled weapons systems, autonomous drones, and next-generation surveillance networks using Nvidia A100 chips sold legally through commercial channels. The A100 is the same chip powering ChatGPT-style large language models. It is also the chip that powers AI-guided weapons targeting systems.

**The Foreign Direct Product Rule (FDPR) extension:** The most aggressive part of the October 2022 rule was the FDPR extension. This rule says: if a foreign-made chip was designed using US EDA software, or if it was manufactured on equipment that contains US-origin technology, then the US government has jurisdiction over where that chip can be sold — even if the chip was made entirely outside the United States.

In practice, because TSMC uses US-origin equipment and software throughout its manufacturing process, the FDPR means the US effectively controls all of TSMC's advanced chip production. TSMC had to comply or risk being cut off from the US technology inputs it needs to operate. The same applies to Samsung and virtually every other advanced foundry.

**The "US persons" ban:** A third element of the rule banned US citizens and permanent residents from supporting Chinese chip fabs with their expertise. This triggered an exodus of US engineers — particularly ethnic Chinese Americans — from positions at SMIC and other Chinese semiconductor firms. Losing this human capital set Chinese chip development back in ways that are difficult to quantify but significant.

**The performance threshold approach:** Rather than banning specific chips by name, the BIS used performance thresholds — defined in terms of total processing performance (TPP) and performance density — to determine which chips require export licenses. This approach was intended to be future-proof, catching new chip generations automatically. Nvidia's response was to design downgraded versions (A800, H800) that stayed just below the threshold. By October 2023, BIS tightened the thresholds to catch those too.

## The Financial Channels: How Controls Hit Markets

When a government issues an export control rule, several financial transmission channels activate simultaneously. Understanding each channel is essential for trading around these events.

**Direct revenue channel:** The most immediate impact is on companies that sell directly to the restricted market. For Nvidia, China was approximately 25-26% of data center segment revenue before the 2022 controls. The mathematical impact is straightforward: revenue from that portion of the market drops to whatever can still be sold legally (lower-performance chips, non-controlled products).

**Supply scarcity channel:** Export controls on advanced chips created an artificial scarcity for the chips that could still be sold freely. When China-bound A100 orders were cancelled, those units did not disappear — they went into a global pool where demand from Western hyperscalers (Microsoft, Google, Amazon, Meta) was already surging due to the generative AI boom. H100 waitlists stretched to 12 months. Pricing power for remaining supply increased dramatically.

**Competitor displacement channel:** Chinese AI companies that could previously buy Nvidia chips had to pivot to domestic alternatives (Huawei's Ascend chips, Biren Technology products) that were substantially less capable. This delayed Chinese frontier AI development, which is a non-trivial competitive advantage for US AI companies.

**Investment redirection channel:** The CHIPS Act (signed August 2022, the same month the export control rules were being finalized) committed \$52 billion in US government subsidies for domestic semiconductor manufacturing. This triggered a construction boom for new US fabs — Intel in Ohio and Arizona, TSMC Arizona, Samsung Texas — that created significant capital expenditure flows and contractor revenue.

**Geopolitical risk premium channel:** Export controls elevated Taiwan Strait tension as a market concern, because the rules underscored how dependent the entire world is on TSMC specifically. Any scenario in which China responds militarily to cross-strait tensions now carries the additional risk of disrupting essentially all advanced chip production globally. This risk premium is embedded in technology sector valuations.

![Timeline of chip war escalation events from 2019 through 2025](/imgs/blogs/semiconductor-export-controls-as-a-geopolitical-weapon-2.png)

## Nvidia's China Revenue Collapse — and the Paradox

The Nvidia story is the most instructive case study in how export controls actually play out in markets, because the outcome confounded almost everyone's expectations.

**Before the controls:** Nvidia's China business was built on two pillars — data center AI chips sold to Chinese tech giants (Alibaba Cloud, Baidu, ByteDance, Tencent) and gaming GPUs sold to Chinese consumers. The data center portion was the high-value business, representing approximately 25-26% of Nvidia's data center segment revenue, or roughly \$4 billion annualized in 2022.

**The workaround attempt:** Nvidia's initial response to the October 2022 controls was to design compliant alternatives. The A800 (data center) and H800 (high-performance computing) were engineered with reduced chip-to-chip interconnect bandwidth — specifically the NVLink bandwidth — to stay below the BIS performance thresholds. These chips were legal to sell in China and represented Nvidia's attempt to preserve the market.

**The tightening:** By October 2023, BIS had closed the loophole. The new rules extended the performance thresholds to cover A800 and H800. Nvidia's legal China-compatible products were effectively eliminated from the advanced AI training market.

**The data center revenue impact:** China's share of Nvidia's data center revenue fell from approximately 26% in FY2022 to roughly 22% in early FY2024, then collapsed to approximately 9% by Q4 FY2024, and was running near 8% by mid-FY2025.

**The paradox:** While China revenue was collapsing, Nvidia's total revenue and stock price were doing the opposite. FY2024 total revenue was \$60.9 billion, up from \$26.9 billion in FY2023. The data center segment alone reached \$47.5 billion. Nvidia's EPS grew from \$1.74 in FY2023 to \$11.93 in FY2024 — a 586% increase. The stock rose approximately 239% in calendar year 2023.

The mechanism was straightforward once you saw it: Western hyperscalers were spending tens of billions of dollars racing to build AI infrastructure, and the export controls — combined with the AI boom — created a situation where H100 chips were so scarce that customers were paying \$30,000-40,000 per unit and waiting 12 months for delivery. The China revenue loss was more than offset by the pricing power that scarcity delivered.

This is the "chip war paradox": losing your largest individual customer can make your business more valuable if it coincides with a demand surge from your remaining customers and a supply ceiling that gives you pricing power.

#### Worked example:

**Nvidia's China revenue impact calculation:**

Nvidia FY2024 total revenue: \$60.9 billion
Data center segment revenue: \$47.5 billion

China data center revenue (pre-controls, 26% of data center):
26% × \$47.5B = \$12.35 billion

China data center revenue (post-controls, 8% of data center):
8% × \$47.5B = \$3.8 billion

Revenue lost to China export controls: \$12.35B − \$3.8B = \$8.55 billion per year

Now, the offset:
- Nvidia EPS grew from \$1.74 (FY2023) to \$11.93 (FY2024): +\$10.19/share gain
- Total shares outstanding: approximately 2.46 billion
- Implied total earnings gain: ~\$25 billion

So Nvidia lost roughly \$8.6 billion in China revenue but gained approximately \$25 billion in total earnings power. The Western AI demand surge effectively absorbed the China loss and more than tripled earnings. This is why the market reacted to the export controls with a stock price rally once the AI capex cycle was understood.

![Nvidia China revenue as percent of data center revenue declining from 26 to 8 percent](/imgs/blogs/semiconductor-export-controls-as-a-geopolitical-weapon-3.png)

![Nvidia stock performance around export control announcements showing initial drop then massive gain](/imgs/blogs/semiconductor-export-controls-as-a-geopolitical-weapon-4.png)

## China's Response: The Self-Sufficiency Push

China's official response to semiconductor export controls was to accelerate its existing plan to achieve domestic chip self-sufficiency — a goal called "Made in China 2025" that had been a stated priority since 2015, but which the 2022 controls turned from an aspiration into an urgent national security imperative.

**The state investment commitment:** China's central government and provincial governments collectively committed over \$150 billion in funding for domestic semiconductor development. The National Integrated Circuit Industry Investment Fund — colloquially called the "Big Fund" — raised its third phase in 2024, targeting roughly 344 billion yuan (approximately \$47 billion) for advanced chip development. This is on top of two earlier phases and extensive provincial-level funding.

**The Huawei Mate 60 Pro moment:** In August 2023, Huawei released the Mate 60 Pro smartphone. When teardown analysts at TechInsights opened it, they found a SMIC-fabricated 7-nanometer chip (the Kirin 9000s). This was a significant jump — SMIC had been estimated to be at 14nm-equivalent capability. The chip demonstrated that China had made meaningful progress in domestic chip manufacturing despite the export controls.

**But context matters enormously:** The SMIC 7nm chip revealed several important limitations that the headlines often missed.

First, yield rates: TSMC's 7nm process runs at approximately 95% yield — meaning 95 out of every 100 chips produced are functional. SMIC's estimated yield rate for its 7nm-equivalent process is around 60%, meaning 40% of production is waste. This has profound cost implications.

Second, scale: TSMC can produce millions of 7nm wafers per month across multiple fabs. SMIC's capacity at 7nm is a fraction of that, constrained by limited access to advanced equipment.

Third, the process generation gap: While China achieved a 7nm capability, TSMC was already volume-producing at 3nm and advancing toward 2nm. The gap at the leading edge actually widened in absolute terms even as China caught up at older nodes.

Fourth, equipment dependency: Even for its 7nm process, SMIC relies on older Deep Ultraviolet (DUV) lithography machines from ASML — the older generation that was still legal to export to China. The Netherlands-Japan DUV ban announced in 2023 has now closed that avenue too, effectively capping SMIC's progress at its current level unless it can develop domestic alternatives (which it is trying to do, but the timeline is long).

#### Worked example:

**Huawei/SMIC 7nm chip cost disadvantage:**

SMIC 7nm estimated yield rate: 60%
To produce 1 functional chip, SMIC must manufacture: 1 ÷ 0.60 = 1.67 chips

TSMC 7nm estimated yield rate: 95%
To produce 1 functional chip, TSMC must manufacture: 1 ÷ 0.95 = 1.053 chips

Assume base wafer production cost is \$2,000 per wafer for both fabs.

SMIC effective cost per good chip: \$2,000 × 1.67 = \$3,340 per wafer equivalent
TSMC effective cost per good chip: \$2,000 × 1.053 = \$2,105 per wafer equivalent

The yield gap alone gives TSMC a roughly 37% cost advantage at 7nm.

But this calculation becomes more severe at 3nm:
- TSMC: volume-producing at \$16,000-\$20,000 per wafer with ~90% yield
- SMIC: not yet capable of producing 3nm chips at any yield

The node gap means China is not just paying a cost premium — it is simply unable to manufacture the chips required for frontier AI training workloads.

## The ASML Factor: The Bottleneck Only One Company Controls

ASML Holding (ASML: AMS, ASML: NASDAQ) is the most important company most investors have never heard of. It is, in a meaningful sense, the linchpin of the entire export control regime.

**What ASML does:** ASML makes lithography machines — the tools that project circuit patterns onto silicon wafers. Its EUV (Extreme Ultraviolet) machines use light with a wavelength of 13.5 nanometers to print features smaller than that. This technology required decades of development and represents one of the most complex pieces of precision engineering in human history. Each EUV machine contains over 100,000 parts, weighs about 180 metric tons, requires two Boeing 747 planes to ship, and costs approximately \$350 million.

**The monopoly:** ASML is the sole maker of EUV machines. Its nearest competitor is Nikon, which makes older DUV machines. There is no credible path for any country — including the US — to develop an alternative EUV source in the next 10-15 years. The physics of EUV lithography, the supply chain for specialized laser sources and reflective optics, and the accumulated process knowledge all reside at ASML.

**The US lever:** ASML's EUV machines contain critical components from US-origin suppliers — including Cymer (a San Diego laser company that ASML acquired) and optical systems with US-origin technology. This means ASML's export licenses are subject to US re-export control authority. When the US pressured the Netherlands to deny China an EUV export license in 2019, ASML could not fight it without risking its entire business model.

**China's ASML exposure:** Prior to 2019, China was ASML's largest single country customer in some periods, representing 27-30% of revenue. China used the older DUV machines for producing legacy chips (28nm and above) — still a massive market for everything from consumer electronics to automotive chips to industrial equipment. The DUV ban extended in 2023 to include the most advanced DUV machines (TWINSCAN NXE and related systems) effectively sealed China's ability to advance beyond its current process node.

**The revenue impact:** ASML's China revenue share fell from approximately 27-30% in 2020-2021 to around 14% in 2022 (as the EUV restrictions took hold), briefly recovered to 26% in 2023 as Chinese customers rushed to stock up on still-legal DUV machines before the expanded ban, then collapsed to approximately 13% in 2024 after the extended DUV restrictions.

This "stock-up" pattern in 2023 ASML's China revenue is itself a trading signal. When export control tightening is anticipated, affected buyers accelerate purchases of whatever is still legal, creating a temporary revenue spike that then reverses sharply.

## The Beneficiaries: Who Wins the Chip War

Export controls, despite their intended role as a restriction, create clear winners alongside the obvious losers. Understanding who benefits is as important for investors as understanding who is harmed.

![Chip war winners and losers matrix showing short-term and long-term positions for major players](/imgs/blogs/semiconductor-export-controls-as-a-geopolitical-weapon-5.png)

**TSMC — the accidental monopolist:** TSMC did not choose to become the world's critical semiconductor infrastructure. But the combination of its technological leadership and the export control regime has made it the sole provider of the most advanced chips to the entire non-Chinese tech economy. The CHIPS Act funded TSMC's Arizona expansion (\$65 billion total investment, with \$11.6 billion in US government grants). TSMC's position is now explicitly protected by both US policy and the fact that there is no credible alternative.

**Nvidia — the scarcity premium story:** As analyzed above, losing China as a customer coincided with a demand environment where remaining customers were desperately competing for supply. The export controls inadvertently gave Nvidia pricing power by removing one large buyer from a market that was supply-constrained.

**Western semiconductor equipment makers:** Companies like Lam Research (etch tools), Applied Materials (deposition tools), KLA Corporation (inspection tools), and Entegris (specialty chemicals) all benefited from the CHIPS Act-driven investment in new US and allied fabs, even as they lost some China revenue.

**Japan's renaissance:** Japan announced a government-backed consortium to build Rapidus, a new foundry in Hokkaido targeting 2nm production by 2027 with IBM process technology transfer. This represents Japan's re-entry into leading-edge chip manufacturing after a decade of withdrawal. The geopolitical chip war gave Japan both a reason and a funding mechanism to make this investment.

**Defense contractors:** Companies with exposure to secure chip supply — including companies building classified computing infrastructure for the US DoD and intelligence community — benefit from a policy environment that prioritizes domestic chip capability regardless of cost.

**The loser side:** SMIC (stuck below 7nm without EUV), Chinese AI startups (competing for expensive domestic GPU alternatives), Chinese cloud hyperscalers (paying a premium for inferior compute), and Huawei (whose smartphone and network equipment businesses are permanently limited by chip access) all face structural disadvantages that compound over time.

#### Worked example:

**The CHIPS Act economics — what it actually costs to bring chips home:**

Intel Arizona fab: total investment approximately \$20 billion
US government support: \$8.5 billion in CHIPS Act direct grants + \$11 billion in loans

Intel's break-even wafer cost in Arizona: estimated at approximately \$5,500 per wafer
TSMC's equivalent wafer cost in Taiwan: approximately \$3,200 per wafer

Cost premium of US domestic production: \$5,500 − \$3,200 = \$2,300 per wafer

At 20,000 wafers per month production capacity:
Monthly "national security premium": \$2,300 × 20,000 = \$46 million
Annual "national security premium": \$46M × 12 = \$552 million per year

US DoD and intelligence community chip procurement: approximately \$10 billion per year

If even 20% of that procurement goes to domestic US-made advanced chips, the annual spend is \$2 billion. Against a \$552 million/year premium, the national security calculus shows a positive return on the subsidy — assuming that domestic supply avoids even one Taiwan-disruption scenario that would otherwise cost tens of billions in defense readiness.

The CHIPS Act grants, viewed this way, are defense spending reframed as industrial policy. The break-even math works on defense procurement alone, before considering the broader economic and technological spillover benefits.

## The Enforcement Gap: How China Works Around Controls

Export controls are not airtight. Any trader or analyst following the chip war needs to understand how enforcement actually works — and where the holes are — because the effectiveness of controls is a key variable in the medium-term outlook.

**Third-country routing:** The most common circumvention method is purchasing chips through third-country intermediaries. A company in Singapore, the UAE, or South Korea legally buys Nvidia chips, then resells them to China. The US Commerce Department's Bureau of Industry and Security (BIS) tracks and prosecutes these cases, but the supply chains are opaque enough that significant volumes slip through.

Singapore re-exports of US semiconductors to China rose from approximately \$1.2 billion in 2021 to \$3.8 billion in 2023, according to trade data cited by Reuters and The Wall Street Journal. Not all of this is circumvention — some is legitimate re-export — but the jump in volume coincided with the implementation of export controls in ways that raised red flags for BIS analysts.

**Cloud compute access:** Chinese companies can still rent access to foreign data centers to run AI workloads on restricted chips, as long as the chips themselves do not enter China. A Chinese AI lab can theoretically rent H100 compute time from a US cloud provider — though US cloud providers have implemented increasingly strict verification procedures under BIS guidance, and the January 2025 AI diffusion rules moved toward closing this channel.

**Below-threshold products:** Not everything Nvidia or AMD makes requires an export license. Consumer gaming GPUs, professional visualization cards, and some networking chips fall below the performance thresholds. These still flow to China freely and represent a residual market for US chip companies.

**Open-source AI models:** Training frontier AI requires massive compute. But inference — running a model once it is trained — requires far less. Open-source models like Meta's Llama family can be run on relatively modest hardware. China's AI researchers are adept at using older hardware efficiently, and the export control gap in inference compute is much smaller than the gap in training compute.

#### Worked example:

**The third-country circumvention arbitrage:**

A Singapore trading company purchases Nvidia A100 GPUs at market price: approximately \$10,000 per unit
The company re-sells to a Chinese buyer willing to pay a scarcity premium: approximately \$30,000 per unit
Arbitrage profit per chip: \$20,000

Estimated volume of circumvented chips reaching China (using Singapore re-export data uplift and BIS prosecution data as proxies): approximately 130,000 chips per year

Total estimated circumvention revenue for intermediaries:
130,000 chips × \$20,000 margin = \$2.6 billion in arbitrage profits per year

BIS enforcement actions in 2023: 29 entities added to the Entity List for circumvention-related violations.

The enforcement gap is real and substantial. But it does not solve China's problem at the frontier — even circumvented A100s are one or two chip generations behind the current H100/H200/B200, so China's AI compute stack remains structurally behind even accounting for smuggling.

## Common Misconceptions

**"The controls will slow down US AI companies too."**

This gets the causation backwards. US AI companies benefit from a domestic chip industry that is not competing to supply China. Export controls, combined with the AI demand boom, tightened supply for all buyers — but US hyperscalers had priority access and could pay any price. The net effect was to slow down Chinese AI development relative to the US, not to slow down both equally.

**"China will just develop its own chips in a few years."**

China will develop better chips — it already has, as the Huawei Mate 60 Pro demonstrated. But "better" is not the same as "competitive with the leading edge." The gap between SMIC's best capability and TSMC's current production is measured in process nodes (SMIC at ~7nm equivalent, TSMC at 2-3nm and advancing). Each node requires not just more advanced lithography but vastly different materials, processes, and yield optimization techniques. The timeline for China to close the leading-edge gap without EUV machines is measured in decades, not years, under any realistic scenario.

**"TSMC moving to Arizona solves the problem."**

TSMC Arizona is a real and important development. But TSMC's Arizona fabs, once complete, will represent a small fraction of TSMC's total capacity — and TSMC's most advanced processes will still be pioneered in Taiwan first. The Arizona fabs are a strategic diversification, not a full relocation. The Taiwan geopolitical risk for semiconductor supply is moderated, not eliminated, by the CHIPS Act investments.

**"Export controls are purely defensive."**

The October 2022 rules were not purely defensive. They were an offensive economic measure intended to degrade China's military AI capability and slow China's ability to compete in AI commercially. The rules represent a strategic decision to use the US position in the semiconductor ecosystem as a coercive tool — the technology equivalent of a naval blockade. Understanding this offensive dimension helps explain why China's response has been so aggressive and so well-funded.

**"Nvidia's China business is just gone."**

Nvidia's China data center business is largely gone for advanced AI chips. But Nvidia still sells gaming GPUs, networking products, and lower-performance AI chips to China under existing licenses. The China business is diminished, not eliminated. And Nvidia has explicitly said it is designing future products to comply with whatever the current export control thresholds are — the cat-and-mouse game between BIS and Nvidia's China product design team continues.

## How It Shows Up in Real Markets

The chip war shows up in several market venues in distinctive ways that traders and investors can monitor.

**Nvidia and AMD earnings calls:** Each quarterly call now includes detailed discussion of export control impacts and China revenue exposure. When BIS tightens thresholds, both companies typically guide lower China revenue on the subsequent call. When new compliant products are announced, it is a signal that the gap between what is legal and what is commercially interesting has narrowed — which is itself a signal that BIS may tighten again.

**ASML's order book:** ASML reports backlog quarterly. The China component of backlog is a leading indicator of how much DUV inventory China is stockpiling before the next control tightening. A surge in China orders at ASML historically precedes a BIS announcement extending controls.

**TSMC capital expenditure guidance:** TSMC's capex guidance reflects both leading-edge capacity investment (always bullish for AI infrastructure) and geographic diversification spending (Arizona, Japan, Germany fabs). A higher proportion of capex going to non-Taiwan fabs signals increasing geopolitical risk management — which is a buy signal for TSMC given that it reflects the company's growing importance to allied governments.

**The Philadelphia Semiconductor Index (SOX):** The SOX is the benchmark equity index for the semiconductor sector. It responds to BIS announcements, Taiwan Strait tension news, and CHIPS Act funding decisions. A BIS tightening announcement typically causes a 2-5% intraday drop in SOX, followed by a recovery as investors process the scarcity premium offset.

**Chinese AI hardware stocks (Hong Kong listed):** Companies like Cambricon Technologies (a Huawei-adjacent AI chip designer) and Hygon Information Technology trade in Hong Kong and reflect Chinese domestic chip capability progress. These names are volatile and thinly traded but serve as proxies for how sophisticated investors assess China's domestic capability gap.

**Taiwan Strait tension indicators:** Any military provocation in the Taiwan Strait — PLA air incursions into Taiwan's Air Defense Identification Zone, live-fire exercises, naval patrols — directly impacts semiconductor sector stocks. TSMC specifically tends to drop 3-7% on serious Taiwan incident headlines, and recovers as the immediate tension passes. The "Taiwan premium" in the semiconductor sector is now a permanent feature of the risk landscape.

![Global semiconductor dependency network showing US chokepoints and China limitations](/imgs/blogs/semiconductor-export-controls-as-a-geopolitical-weapon-6.png)

## How to Trade It / The Playbook

Understanding the chip war framework gives investors a playbook for responding to specific events. Here are the core scenarios and their implications.

**Scenario 1: BIS announces tightened export control thresholds.**

*Immediate reaction:* Nvidia and AMD drop 5-15% as markets price in additional China revenue loss. ASML drops 5-10% on reduced China demand expectations. SOX drops 3-6%.

*The trade:* Buy the dip in Nvidia and AMD within 5-10 trading days if the tightening is in response to documented circumvention. The pattern from October 2022 and October 2023 is clear — the initial drop is an overreaction. The scarcity premium from lost China supply accrues to Western AI buyers who need the chips regardless of price. Both companies outperformed following both tightening events.

*Avoid:* Companies with genuine China-only revenue exposure (no Western AI offset) — these do not recover on the same timeline.

**Scenario 2: China announces domestic chip breakthrough.**

*Immediate reaction:* Markets tend to over-react to China chip news. The Huawei Mate 60 Pro caused NVDA to drop approximately 5% in two days. This response is typically overdone.

*The trade:* A domestic China chip capability at 7nm is not a substitute for H100-class compute for AI training. Sell the pop in Chinese AI hardware stocks (which may initially rally on domestic champion narratives) and hold Western chipmakers through the noise. The gap between 7nm and 2nm is far larger than the gap between 14nm and 7nm in terms of AI training capacity.

*Watch:* If China announces genuine EUV capability (i.e., domestically developed EUV machines with competitive output), that is a regime change. Currently this is not on a credible 5-year horizon.

**Scenario 3: Taiwan Strait military incident.**

*Immediate reaction:* TSMC drops 10-20%. The entire semiconductor sector drops 5-15%. Safe haven flows into gold and US Treasuries.

*The trade:* This is the hardest scenario because it depends on severity. A limited PLA exercise (as happened in August 2022 after Speaker Pelosi's visit) typically resolves within 2-3 weeks and the chip sector recovers fully. A genuine blockade or military assault is a multi-year disruption with no historical precedent to use for recovery timing.

*Position sizing:* Given the asymmetric downside in the Taiwan tail risk, many institutional investors run a small permanent hedge in semiconductor sector puts or TSMC puts — the cost of the hedge is considered a legitimate "insurance premium" for the Taiwan concentration risk.

**Scenario 4: CHIPS Act funding announcement for specific fab.**

*Immediate reaction:* Recipients (Intel, TSMC Arizona, Samsung Texas) pop 3-8%. Suppliers to those fabs (Applied Materials, Lam Research, KLA) pop 2-5%.

*The trade:* These pops are real. Announced funding creates a clear revenue line for construction and equipment procurement that translates into earnings within 2-4 years. The capital allocation signals are clear and well-publicized. These are momentum events for the supply-chain beneficiaries.

**Scenario 5: Election-driven policy reversal signals.**

Export controls are currently bipartisan in the US — the October 2022 Biden rules were extended and tightened, and both Republican and Democratic administrations have maintained strong semiconductor control policy. However, any signal of rollback (e.g., hypothetical trade deal components that relax controls) would be a significant negative for the "scarcity premium" thesis for Nvidia and a positive for ASML's China revenue.

*The trade:* If credible policy reversal signals emerge, rotate from Nvidia into ASML. ASML's China revenue recovers faster than Nvidia's competitive moat would adjust. ASML has a hardware product the world needs regardless of geopolitics; Nvidia competes in a market where China recovering supply could increase competition.

![ASML revenue by region 2020 to 2024 showing China share declining from 27 to 13 percent](/imgs/blogs/semiconductor-export-controls-as-a-geopolitical-weapon-7.png)

#### Worked example:

**The ASML China stock-up trade:**

In Q3 2023, ASML's China revenue surged to approximately 26% of total (from 14% in 2022), as Chinese customers rushed to purchase DUV machines before the extended ban took effect. ASML's quarterly earnings in this period beat consensus estimates significantly.

A trader who understood the export control escalation timeline could have anticipated this stock-up dynamic:

- January 2023: Netherlands announces intent to restrict advanced DUV exports
- Spring 2023: Chinese customers begin large DUV purchase orders to build inventory before restrictions
- Q3 2023 ASML earnings: China revenue beats consensus by approximately 15%
- ASML stock reaction: +8-12% in the quarter following the earnings beat

The trade was to buy ASML before the Q3 2023 earnings announcement, knowing that the restriction announcement had set off a stock-up race by Chinese customers. Post-earnings, the position could be lightened with the understanding that Q4 2023 and 2024 China revenue would decline sharply once the rush buying was complete.

This "anticipate the stock-up, sell the implementation" pattern is repeatable. Any time a major export control extension is announced with a 6-12 month implementation window, the affected manufacturer's China customers will rush to buy what they can before the window closes.

## The Dutch and Japanese Alignment: How the US Built a Coalition

The October 2022 BIS rule had a fatal weakness if it remained a unilateral US action: China could simply buy the equipment it needed from non-US suppliers. The reason the export control regime actually works is that the US convinced the two other countries that matter — the Netherlands and Japan — to align their own export policies. Together, these three nations control the chokepoint equipment for every semiconductor process node below 10nm. No fourth country can fill the gap.

**Why ASML matters more than any other single company:** The Netherlands hosts ASML, the only company on earth that builds Extreme Ultraviolet (EUV) lithography machines. EUV is not optional for sub-7nm chips — it is the only commercially viable way to print features that small. If China cannot buy EUV machines, it cannot manufacture leading-edge logic chips, full stop. This makes the Dutch government's export policy the single most important lever in the entire control regime, more important even than US policy on Nvidia chips, because EUV controls the upstream capability rather than the downstream product.

**The four-year pressure campaign on the Dutch:** The US did not get Dutch cooperation overnight. Washington first persuaded the Netherlands to withhold ASML's EUV export license to China in 2019 — a quiet administrative action rather than a formal ban. It took until 2023 for the Netherlands to formalize and broaden these restrictions into law, extending them to cover the most advanced DUV machines as well. That four-year timeline tells you how much diplomatic capital the US had to spend: the Netherlands was being asked to forgo billions in revenue at a domestic crown-jewel company, and Dutch policymakers resisted until the security argument and US leverage over ASML's US-origin components made compliance unavoidable.

**Japan's October 2023 controls:** Japan formalized its alignment in 2023 by adding 23 categories of semiconductor manufacturing equipment to its export restriction list. This was the decisive third leg. Japanese firms like Tokyo Electron (deposition, etch, and cleaning equipment), Shin-Etsu Chemical (silicon wafers and photoresists), and Sumco (wafers) supply inputs that China cannot source domestically at the required quality. The 23-category restriction meant that even the equipment China could not get from ASML, it now also could not get from Tokyo Electron — closing the most obvious workaround.

**Why the trilateral coalition is uniquely effective:** The power of the US-Netherlands-Japan bloc is that the three countries together cover the entire equipment stack for advanced nodes. ASML covers lithography, Tokyo Electron and Applied Materials and Lam Research cover deposition and etch, and Japanese chemical firms cover the materials. There is no node below 10nm that can be manufactured without touching equipment or materials from at least one of these three countries. A bilateral US-only regime would leak; the trilateral regime does not.

**Market impact of the coalition news:** Markets repriced these companies sharply on the control announcements. ASML shares fell approximately 5.3% on news that its China sales would face restrictions, as investors marked down a chunk of forward revenue. Tokyo Electron fell about 7% on the announcement of Japan's controls, reflecting that roughly a quarter of its revenue came from Chinese fabs that would now face restricted purchasing. These were not panic moves — they were rational repricings of the discounted China revenue stream that the controls removed.

#### Worked example:

**Calculating ASML's China revenue exposure:**

ASML 2022 total revenue: approximately \$19.2 billion (21.2 billion euros at the period exchange rate)
ASML 2022 China revenue: approximately \$2.3 billion

China share of total revenue: \$2.3B ÷ \$19.2B = 12.0%

Now separate the EUV restriction from the DUV restriction. China was already banned from EUV since 2019, so the 2022 China revenue was almost entirely DUV and older systems. The 2023 extension of controls to advanced DUV is what threatened this remaining stream.

Estimated DUV revenue at risk from the extended restrictions: approximately \$1.4 billion annually (the portion of China DUV sales falling into the newly restricted advanced categories, with mature-node DUV still permitted).

Stock price reaction math on the 5.3% decline:
ASML market capitalization at the time: approximately \$260 billion
5.3% decline implies a market-cap loss of: 0.053 × \$260B = \$13.8 billion

The market wiped roughly \$13.8 billion off ASML's value over a revenue stream worth about \$1.4 billion per year. At a steady-state, that implies the market was capitalizing the lost China DUV revenue at roughly 10 times annual sales — consistent with treating the China advanced-DUV business as a permanently impaired growth line rather than a one-year revenue dip. This is the signature of how markets price export controls: they discount the entire forward stream, not just the next quarter.

## China's Response: The \$150 Billion Self-Sufficiency Plan

Faced with a coordinated wall of equipment restrictions, China did what it does best: it threw enormous amounts of state capital at the problem. The scale of the commitment is staggering, but the results so far reveal a hard truth — at the leading edge, the binding constraint is physics and accumulated know-how, not money.

**The Big Fund:** China's primary vehicle is the National Integrated Circuit Industry Investment Fund, universally called the "Big Fund." It has been deployed in three phases. Fund I (2014) raised approximately \$48 billion. Fund II (2019) raised approximately \$29 billion. Fund III (2024) is reported at roughly \$40-47 billion, the single largest phase, explicitly oriented toward equipment and advanced manufacturing in response to the export controls. Across all three phases plus provincial-level matching funds, the total state commitment exceeds \$150 billion.

**Why chips are not solar panels:** China has a playbook for dominating an industry: subsidize aggressively, build overwhelming capacity, drive Western competitors out on price. It worked spectacularly for solar panels, where China now controls the global supply chain. But that playbook fails at the leading edge of semiconductors because the constraint is not capital or capacity — it is the EUV machine that no amount of money can buy, and the decades of process engineering that no amount of money can buy quickly. You cannot subsidize your way past a physics barrier when the one tool that crosses it is export-banned.

**The technology gap:** China's most advanced fab, SMIC, is stuck at roughly 7nm and limited to the N-1 generation — it can produce 7nm chips, but only with multiple-patterning workarounds on older DUV equipment, at low yield and limited volume. Meanwhile TSMC is shipping 2nm. That is not one generation of difference; it is three to four full nodes, and each node is harder than the last.

**The Kirin 9000s: the exception that proves the rule:** When Huawei shipped the Kirin 9000s in its 2023 Mate 60 Pro, headlines proclaimed that China had beaten the export controls. The reality was more sobering. The chip was real and was produced at SMIC's 7nm process — a genuine achievement. But it reportedly cost around three times what TSMC charges for an equivalent node, and SMIC's yield rates on the process were an estimated 30-40% lower than TSMC's. A chip you can make at triple the cost and a fraction of the yield is a proof of concept, not a competitive commercial product. It demonstrated capability while simultaneously demonstrating the economic gap.

**The talent constraint:** Perhaps the most underappreciated barrier is human capital. TSMC employs roughly 70,000 engineers in Taiwan, many with decades of node-transition experience that exists nowhere else. China cannot replicate that depth of accumulated expertise in a decade, and the October 2022 "US persons" rule made it worse by forcing US-citizen engineers to leave Chinese fabs. Money builds cleanrooms; it does not instantly create the institutional knowledge that lives in the heads of tens of thousands of veteran process engineers.

**What China can actually do:** The self-sufficiency push is not a failure — it is simply succeeding in the places where physics permits. China can and does win in advanced packaging (stacking and interconnecting chips, where it is competitive), in memory (YMTC's NAND flash has reached respectable density), and in mature nodes (28nm and above, which power the vast majority of industrial, automotive, and consumer electronics). These are real markets worth tens of billions of dollars, and Chinese dominance of mature-node capacity creates both oversupply risk for legacy chipmakers and investment opportunities in the suppliers feeding China's mature-node buildout.

![China domestic semiconductor investment timeline and technology gap versus TSMC leading edge](/imgs/blogs/semiconductor-export-controls-as-a-geopolitical-weapon-8.png)

## Trading the Chip War: Equipment Stocks, Memory Plays, and the Long Game

The chip war is not a single trade. It is a structural realignment that creates distinct opportunities and traps across the equipment, memory, and logic layers of the supply chain. The mistake most investors make is treating "China restriction news" as uniformly bearish for chip stocks. The reality is more nuanced, and the nuance is where the edge lives.

**The equipment supplier exposure map:** The US semiconductor equipment makers all have material China revenue, but the proportions differ and matter. KLA Corporation (process control and inspection tools) derives roughly 20% of revenue from China. Lam Research (etch and deposition) is the most China-exposed, at roughly 30%. Applied Materials (the broadest equipment portfolio) sits around 26%. On any China-restriction headline, these three sell off together — but the magnitude of justified selloff scales with the exposure, and the market often paints them with the same brush, creating relative-value dislocations.

**Why equipment stocks are not a clean short:** The intuitive trade on tightening controls is to short the equipment makers losing China revenue. This has repeatedly burned traders. The reason is pre-restriction stockpiling: Chinese fabs, knowing more restrictions are coming, buy legacy-node and still-permitted equipment as fast as they physically can. This pulls forward revenue and props up the equipment makers' near-term results even as the long-term China opportunity shrinks. A short thesis built on "China revenue is going away" can be wrong for six to twelve months while the stockpiling runs.

**The long side:** The cleaner trades are on the long side. TSMC captures market share as Chinese fabs fall further behind at the leading edge — every node TSMC advances while SMIC stalls is share that flows to Taiwan. Tokyo Electron commands a premium as the non-Chinese alternative for fabs that need to derisk their equipment supply away from any China entanglement. Entegris, which supplies the advanced materials and filtration the most cutting-edge nodes depend on, benefits from the leading-edge buildout regardless of geography. These names ride the structural tailwind rather than fighting the stockpiling noise.

**The memory bifurcation:** Memory is splitting into two worlds. YMTC, China's NAND flash champion, has made genuine progress but its advanced-node NAND remains an estimated 18-24 months behind the leaders. The leaders — Samsung, SK Hynix, and Micron — hold the high-margin, leading-density segment. The investment implication is that the memory market is bifurcating into a Chinese commodity tier and a Western/Korean advanced tier, and the advanced tier retains pricing power precisely because YMTC cannot yet compete there. Long the advanced-tier names; treat YMTC's progress as a slow grind that pressures only the commodity segment.

**The asymmetric trade:** The cleanest structural expression of the whole thesis is to be long US advanced-logic capability and short the legacy semiconductor commodity. SMIC and its peers are being funneled by the export controls into exactly the place where they can compete — the mature 28nm node — which happens to be the most oversupplied segment of the global chip market. As Chinese state capital floods 28nm capacity, the legacy-node commodity faces margin compression, while advanced logic (TSMC, Nvidia's designs) faces scarcity-driven pricing power. Long the scarce, short the glut.

#### Worked example:

**Long NVDA / Short AMAT relative value trade around the October 2022 BIS announcement:**

On the October 7, 2022 BIS announcement, both names sold off, but for different reasons. Nvidia dropped on lost China chip revenue; Applied Materials dropped on lost China equipment revenue.

Entry on the news (approximate moves from the announcement):
NVDA: approximately −20% in the weeks following the rule
AMAT: approximately −15% in the same window

Set up the ratio trade: long \$1,000,000 NVDA, short \$1,000,000 AMAT, dollar-neutral at entry.

The thesis: Nvidia's China loss is offset by the Western AI demand surge and a supply-constrained scarcity premium, while Applied Materials faces a genuine structural China headwind with weaker offset because its growth was more China-weighted.

Path to end of 2023:
NVDA recovered and then ran to approximately +239% from the entry-area lows
AMAT recovered approximately +40% over the same period

Relative-value return on the ratio:
Long leg: +239% on \$1,000,000 = +\$2,390,000
Short leg: −40% on the \$1,000,000 short (the short loses money as AMAT rises) = −\$400,000
Net ratio P&L: \$2,390,000 − \$400,000 = +\$1,990,000

The spread between the two names widened by roughly 199 percentage points over the period. The trade captured the core insight of the chip war: the export controls handed Nvidia a scarcity premium that dwarfed its China loss, while the equipment makers faced a real China revenue impairment with a thinner offset.

When to close: the natural exit was when Nvidia's China revenue had fully washed out of the comparison base (by Q4 FY2024, China was down to roughly 8-9% of data center revenue) and the AI capex narrative was fully priced. Holding the ratio past that point meant betting on continued multiple expansion rather than on the structural export-control thesis, which had by then played out.

## Further Reading and Cross-Links

The semiconductor chip war does not exist in isolation — it is one front of a broader geopolitical and market conflict between the US and China. To build a complete picture, these related analyses provide essential context.

For the Taiwan dimension of semiconductor risk — including how markets price the specific risk of a Taiwan Strait military incident disrupting TSMC's production — see [Taiwan Strait Tensions, Semiconductor Risk, and Supply Chain Repricing](/blog/trading/geopolitical-crises/taiwan-strait-tensions-semiconductor-risk-and-supply-chain-repricing). That post covers the military scenario analysis and how to hedge Taiwan-specific semiconductor concentration risk.

For the broader framework of how geopolitical events get priced into markets before, during, and after the event — including how the chip war escalation fits into the general framework of "geopolitical risk premium" — see [Geopolitical Risk Premium: What Markets Price In](/blog/trading/geopolitical-crises/geopolitical-risk-premium-what-markets-price-in).

When semiconductor sector selloffs trigger broader risk-off moves, capital tends to flow into traditional safe havens: US Treasuries, gold, the yen, and the Swiss franc. Understanding how these flows work is essential for portfolio management around geopolitical events. See [Safe Havens, Flight to Quality, and the Dollar in a Crisis](/blog/trading/geopolitical-crises/safe-havens-flight-to-quality-and-the-dollar-in-a-crisis).


The chip war is a story about power, technology, and economic interdependence colliding at a moment when AI has made computational capability into national security infrastructure. The October 2022 BIS rule was not just a trade restriction — it was the opening act of a technological decoupling that will shape the global technology industry for decades.

For investors, the key insight is that this decoupling is not bad for the semiconductor sector in aggregate. It is a reallocation. Capital is flowing to domestic manufacturing, to equipment makers supplying new fabs, and to the US chip designers who have effectively been handed a Western-market monopoly on the most advanced AI compute. The companies on the right side of the control regime — TSMC (for now), Nvidia, ASML (in the non-China world), and the CHIPS Act beneficiaries — are structurally advantaged by a policy environment that is bipartisan in the US, multilateral with allied nations, and unlikely to reverse unless the underlying US-China geopolitical competition itself reverses.

The chip war is not ending. It is accelerating. The question for markets is not whether to have exposure to the semiconductor sector, but which part of the global chip supply chain benefits from the new geography of technology power.
