---
title: "Sum-of-Parts Valuation in Vietnam: The Vingroup Case Study"
date: "2026-06-27"
publishDate: "2026-06-27"
description: "How to value a conglomerate like Vingroup by pricing each business separately, subtracting debt, and applying a conglomerate discount."
tags: ["valuation", "asset-pricing", "sum-of-parts", "sotp", "vingroup", "vietnam-stocks", "conglomerate-discount", "ev-revenue", "ev-ebitda", "case-study"]
category: "trading"
subcategory: "Finance"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — When one company owns several very different businesses, you cannot price it with a single multiple; you price each piece on its own terms, add the pieces, subtract the parent's debt, and then knock off a discount because the market distrusts complexity.
>
> - Sum-of-Parts (SOTP) values a conglomerate by valuing each division separately, then summing the stakes the parent actually owns.
> - Different divisions demand different methods: Vinhomes by EV/EBITDA, VinFast by EV/Revenue (it has no profit yet), Vincom Retail by a property cap rate.
> - A 15-25% "conglomerate discount" is normal in emerging markets, and it is larger when one family controls the group and subsidiaries cross-guarantee the parent's debt.
> - The single number to remember: Vingroup's gross sum-of-parts works out near 298,900 billion VND, but its market cap sits around 180,000 billion VND — that gap *is* the conglomerate discount in action.

In August 2023, a Vietnamese electric-vehicle maker called VinFast listed on NASDAQ and, for a few surreal days, carried a market value above 90 billion US dollars — briefly worth more than Ford and General Motors combined, despite selling a tiny fraction of their cars. By 2024 that number had collapsed toward 10 billion dollars. VinFast's parent company, Vingroup, owned roughly three-quarters of it. So what, exactly, was Vingroup worth while its largest single asset swung by a factor of nine?

You cannot answer that with a price-to-earnings ratio. Vingroup is a holding company: a parent that owns chunks of a real estate developer, a carmaker, a chain of shopping malls, hospitals, schools, and a bus company. These businesses have nothing in common. One throws off steady cash from selling apartments. One burns cash building factories and has never earned a profit. One collects rent like a landlord. Multiply Vingroup's blended earnings by some average multiple and you get a number that means nothing, because the earnings themselves are a meaningless blend.

The honest way to value a business like this is **Sum-of-Parts**, usually shortened to **SOTP**. You take the company apart, value each piece using the method that fits that piece, add up only the slices the parent actually owns, subtract what the parent owes, and then — because markets reliably distrust sprawling conglomerates — you take a haircut. This post builds the method from zero and then walks the full Vingroup calculation, number by number, so that by the end you can pick up any conglomerate's annual report and value it the same way.

![Vingroup corporate structure showing VIC parent and its ownership stakes in subsidiaries](/imgs/blogs/sotp-valuation-vietnam-vingroup-case-study-1.png)

## Foundations: why conglomerates break single-multiple valuation

Before any arithmetic, a few terms, each defined in plain language with an everyday-money comparison.

A **multiple** is a shortcut for value. If a corner shop earns 100 million VND a year in profit and similar shops change hands for "ten times earnings," the shop is worth about 1 billion VND. The "ten times" is the multiple. The most famous one is the **price-to-earnings ratio (P/E)**: the company's stock price divided by its profit per share. A P/E of 15 means buyers are paying 15 VND for every 1 VND of annual profit. Multiples are wonderfully fast, and they work beautifully when you are comparing two businesses that look alike — two apartment developers, two banks, two soda companies.

They fall apart the moment the thing you are valuing is *not* one business but several glued together. Here is the everyday version. Suppose your neighbor owns three things: a rental apartment that pays steady monthly rent, a food stall that just opened and is still losing money while it builds a customer base, and a vintage motorbike collection. If a friend asked "what is your neighbor's net worth, as a multiple of his current income?" you would refuse the question. The motorbikes produce no income at all but have real resale value. The food stall has negative income but might be worth a lot if it succeeds. The rental has steady income you *can* multiply. The only sane answer is to price each thing the way that thing should be priced, then add them up. That is Sum-of-Parts.

A **conglomerate** is the corporate version of that neighbor: a single listed company that owns several distinct businesses, often in unrelated industries. A **holding company** (or "HoldCo") is the legal parent at the top that holds the shares of the operating businesses below it. When you buy one share of the parent, you are buying a fractional claim on every business underneath — but only in proportion to how much of each the parent owns.

That ownership proportion is the **stake**. If the parent owns 69% of a subsidiary, then 69% of that subsidiary's value belongs to the parent's shareholders; the other 31% belongs to outside shareholders and is *not yours*. This is the single most common beginner error in conglomerate valuation: counting 100% of a subsidiary's value when the parent only owns a slice.

Two more terms you will need throughout:

- **Enterprise Value (EV)** is the value of the whole operating business regardless of how it is financed — the value to *all* providers of capital, both shareholders and lenders. **Equity value** (also called market cap for a listed firm) is the slice that belongs to shareholders alone. The bridge between them is debt: roughly, `Equity value = Enterprise Value − net debt`, where net debt is total borrowings minus cash. We lean on this bridge constantly. If you want the longer treatment, see [enterprise value vs market cap](/blog/trading/asset-valuation/enterprise-value-vs-market-cap-implied-growth-rates).
- **EBITDA** stands for earnings before interest, taxes, depreciation, and amortization. It is a rough proxy for the cash a business generates from operations before financing and accounting choices muddy the picture. People multiply EBITDA (the "EV/EBITDA multiple") when they want to compare operating businesses without being distracted by differing debt loads.

The reason single multiples fail for conglomerates is not subtle once you see it. A blended P/E hides everything interesting. It hides the fact that one division is growing 40% a year while another is shrinking. It hides the fact that one division has no profit at all to multiply. It produces one average number that describes no actual business inside the company. SOTP exists precisely to undo that blending.

### The conglomerate discount

There is one more foundational idea, and it is the most counterintuitive: a conglomerate is usually worth *less* than the sum of its parts, even though arithmetic says the whole should equal the sum. This persistent gap is the **conglomerate discount**.

If you add up what each Vingroup division would be worth as a standalone listed company, and then compare that to Vingroup's actual market value, the actual value is reliably lower — often 15% to 25% lower in emerging markets, and sometimes more. The market is not making an arithmetic mistake. It is pricing in a set of real frictions that we will catalog in detail later: the difficulty of seeing inside a complex structure, the risk that the controlling family moves cash between divisions in ways that do not benefit outside shareholders, the danger that healthy subsidiaries are pledged to guarantee the parent's debt, and the simple fact that capital trapped inside a sprawling group is harder to get back out.

Understanding both halves — that SOTP gives you the gross sum, and that a discount must then be applied — is the whole game.

### Why "stake" is the trap that catches everyone

It is worth slowing down on the stake, because it is where careful arithmetic most often goes wrong, and the error is always in the same direction: overvaluation. When you read a headline like "VinFast is worth ten billion dollars," that number is the value of *all* of VinFast's equity. Vingroup does not own all of VinFast. It owns about 73%. The remaining 27% belongs to other shareholders — early investors, public buyers on NASDAQ, employees with stock. Those people have a real, legal claim on VinFast that has nothing to do with Vingroup's shareholders.

So the value that flows up to a Vingroup share is not VinFast's full value; it is 73% of it. The 27% is called the **minority interest** (or non-controlling interest), and it is a leak in the pipe between the subsidiary's value and the parent's value. Every division has its own leak: 31% of Vinhomes leaks away, 48% of Vincom Retail leaks away. A SOTP that forgets these leaks counts money that does not belong to the parent's owners, and the overcount can run into tens of trillions of dong.

There is a mirror-image subtlety on the way down, too. The consolidated financial statements of a parent like Vingroup typically *include* 100% of a controlled subsidiary's revenue and assets, then back out the minority's share lower down as "minority interest." This is why you cannot read a conglomerate's consolidated income statement at face value: the revenue line may include sales from a business the parent only 52% owns. SOTP sidesteps this confusion entirely by valuing each subsidiary on its own and then explicitly multiplying by the stake — no consolidation games, no buried minority adjustments.

### A simple cash-flow analogy for the discount

Why would a bundle of valuable things sell for less than the sum of its parts? An everyday version makes the mechanism concrete. Suppose a landlord owns three apartments worth 1 billion VND each, so 3 billion in total, and offers to sell you not the apartments but a *share in a company* that owns all three. You would pay less than 3 billion for that share — and rationally so. You cannot pick which apartment to sell if you need cash. You cannot stop the landlord from spending one apartment's rent renovating another that you think is a money pit. You cannot easily see each apartment's true rent because they are reported as one lump. And if the landlord borrowed against the best apartment to fund the worst, your claim on the good one is encumbered. Every one of those frictions is a reason to pay less than 3 billion. The sum of the parts is real; your *access* to it is impaired. That impaired access is the conglomerate discount, and it is exactly what Vingroup's outside shareholders face.

## The SOTP method, step by step

Sum-of-Parts is a five-step recipe. None of the steps is hard; the discipline is in doing each one honestly and not skipping the unglamorous parts (debt, stakes, discount). Most botched conglomerate valuations are not wrong because the analyst chose a bad multiple; they are wrong because the analyst skipped a step — forgot to scale by the stake, forgot the parent's debt, or forgot the discount entirely — and so produced a clean-looking number built on a missing subtraction. The recipe below is deliberately mechanical precisely so that none of those subtractions can be silently dropped.

1. **Enumerate the divisions.** List every business the parent owns, listed and unlisted. Note the parent's ownership stake in each.
2. **Value each division separately**, using the method that fits its cash-flow profile. A steady cash generator gets an EV/EBITDA or DCF treatment; a pre-profit growth business gets EV/Revenue; a property-rental business gets a cap-rate or REIT-style treatment.
3. **Scale each division's value by the parent's stake.** Take only the slice the parent owns.
4. **Sum the stakes, then subtract holding-company items**: the parent's own net debt and any unallocated head-office costs. This gives the gross SOTP equity value.
5. **Apply a conglomerate discount** to reach the value the market is likely to pay, then divide by shares outstanding for a per-share figure.

The bridge from a pile of segment values down to a per-share number is worth picturing as a stack, because the subtractions are where beginners lose value they should have captured (or, more often, forget to subtract and overvalue the company).

![Before and after comparison of single P/E valuation versus SOTP for Vingroup](/imgs/blogs/sotp-valuation-vietnam-vingroup-case-study-3.png)

The figure above contrasts the two roads. On the left, the single-multiple road dead-ends: there is no sensible profit to multiply once VinFast's losses are mixed in. On the right, each division is priced on its own terms and the pieces add up to something defensible.

It is worth being precise about what "value each division separately" really demands, because the phrase hides the hardest judgment in the whole method: choosing the multiple or model. Each division's value comes from one of three families of method, and which family fits depends entirely on where the business is in its life and how it makes money.

The first family is **earnings-based multiples** — EV/EBITDA and P/E. These work when a business generates stable, positive operating cash you can multiply. A mature property developer or an industrial manufacturer fits here. The multiple you choose should come from comparable listed companies in the same industry and geography: an emerging-market developer trades at a different EV/EBITDA than a developed-market one, and pretending otherwise imports the wrong risk premium.

The second family is **revenue-based multiples** — EV/Revenue. These are the fallback when a business has sales but no profit yet, usually because it is young and reinvesting everything into growth. A pre-profit electric-vehicle maker fits here. The multiple embeds a bet about future margins, which is why it is so volatile: a small change in the market's belief about whether the business will *ever* be profitable swings the multiple enormously. This is the single riskiest method in any SOTP, and we will see it dominate the Vingroup answer.

The third family is **asset-based methods** — cap rates for property, net asset value (NAV) for resource or investment businesses, and discounted cash flow (DCF) for anything whose cash you can forecast. A shopping-mall operator is valued like the buildings it owns, using a cap rate on its rent. An unlisted hospital chain with predictable patient revenue is valued by DCF. These methods anchor to the underlying assets or to forecast cash, not to a peer's trading multiple, which makes them more defensible but more labor-intensive.

The discipline of SOTP is matching each division to the right family and then resisting the temptation to use a single convenient multiple across all of them.

The other half of getting SOTP right is choosing the correct method for each kind of business. There is no universal multiple. A carmaker that loses money cannot be valued on earnings; a mall operator should be valued more like a building than like a manufacturer. The table below maps business types to the method that fits.

![Grid table mapping each Vingroup division to its stake, valuation method, and value to VIC](/imgs/blogs/sotp-valuation-vietnam-vingroup-case-study-6.png)

Now let us actually value Vingroup, division by division. All figures are 2024 estimates from SSI Research; treat them as illustrative orders of magnitude, not audited accounts. Throughout, I use an exchange rate of roughly 25,000 VND per US dollar.

## Valuing Vinhomes: the EV/EBITDA approach

Vinhomes (ticker VHM on the Ho Chi Minh exchange) is Vietnam's largest listed residential real estate developer — the business that builds and sells the apartment townships you see ringing Hanoi and Ho Chi Minh City. It is the steady engine of the group: it actually earns profits and generates cash, which means we can value it on its operating earnings.

The right tool here is **EV/EBITDA**. We estimate the division's EBITDA, multiply by a sensible multiple for an emerging-market property developer (typically 8 to 12 times), arrive at Enterprise Value, then bridge to equity value by subtracting the division's net debt, and finally take Vingroup's ownership slice. For the deeper mechanics of property-specific valuation, see [real estate valuation: cap rate, NOI, DCF](/blog/trading/asset-valuation/real-estate-valuation-cap-rate-noi-dcf).

#### Worked example: Vinhomes at 10x EV/EBITDA

Start with the inputs. Vinhomes' 2024 estimated EBITDA is about 25,000 billion VND. Emerging-market developers trade around 8-12x; we pick the midpoint, 10x.

Step 1 — Enterprise Value: `EV = EBITDA × multiple = 25,000B × 10 = 250,000B VND`.

Step 2 — bridge to equity value. Vinhomes carries modest net debt at the subsidiary level; assume it nets to roughly zero against its large land-bank cash for this illustration, so equity value ≈ EV ≈ 250,000B VND. (In a real model you would pull the exact net debt off the balance sheet; here it is small relative to EV.)

Step 3 — take Vingroup's stake. VIC owns about 69% of Vinhomes: `0.69 × 200,000B ≈ 138,000B VND`. Note the subtle point: the *market* values all of Vinhomes' listed equity near 200,000B VND, and our 10x EBITDA estimate (250,000B) is a touch above that, which simply means our multiple is mildly optimistic versus where the stock actually trades. For the SOTP we anchor to the observable market cap of 200,000B and take VIC's 69% slice, giving **138,000 billion VND** attributable to Vingroup.

The intuition: Vinhomes is the one division you can value the "normal" way, on its earnings, and it alone accounts for roughly three-quarters of a trillion VND of value before we even reach the controversial parts.

A note on why we used the listed market cap rather than our own 10x estimate: when a subsidiary is *already publicly traded*, its market cap is a live, observable price for 100% of its equity. That is usually a better anchor than your own multiple guess, because the market is doing the valuing for you in real time. Your EV/EBITDA estimate is most useful as a sanity check (is the stock cheap or dear versus fundamentals?) and as the *only* available tool when a division is unlisted.

### An alternative for developers: price-to-NAV

There is a second, arguably better, way to value a property developer like Vinhomes: **price-to-net-asset-value (P/NAV)**. A developer's real worth is the value of its land bank and projects, marked to current prices, minus the debt against them. That marked value is the **net asset value (NAV)**. Analysts then ask what multiple of NAV the stock trades at. A developer trading at 1.0x NAV is priced exactly at the appraised value of its assets; below 1.0x, the market is skeptical the land will be developed profitably; above 1.0x, the market expects the developer to create value beyond the raw land.

For Vinhomes, the P/NAV lens matters because so much of its value sits in an enormous land bank acquired years ago at low cost. An EV/EBITDA multiple captures the *flow* of current project sales; a P/NAV captures the *stock* of land waiting to be developed. In practice an analyst would compute both and reconcile them. If EV/EBITDA gives 250,000B and a NAV appraisal gives, say, 220,000B, the gap tells you how much of the developer's value is current cash flow versus banked land — and which assumption your final number leans on. The cap-rate and NAV machinery for property is treated in full in the [real estate valuation guide](/blog/trading/asset-valuation/real-estate-valuation-cap-rate-noi-dcf).

The broader lesson: even within a single division you usually have two or three defensible methods, and a serious valuation triangulates among them rather than trusting one. When the methods agree, you gain confidence; when they diverge, the divergence itself is information about where the value is hiding.

## Valuing VinFast: EV/Revenue for a pre-profit company

VinFast (ticker VFS on NASDAQ) is the hard one, and it is where most naive valuations of Vingroup go wrong. VinFast makes electric vehicles. It is growing fast, spending enormous sums on factories and dealerships, and — crucially — it has never turned a profit. It has *negative* EBITDA. You cannot multiply a negative number by a multiple and get anything meaningful.

When a company has revenue but no profit, the standard tool is **EV/Revenue**: value the enterprise as a multiple of its sales rather than its (nonexistent) earnings. The bet embedded in an EV/Revenue multiple is that today's sales will eventually convert into tomorrow's profits. Comparable pre-profit or thin-profit EV makers — NIO, Li Auto, Rivian, Lucid — have traded across a wide band, roughly 0.5x to 2x revenue depending on the market's appetite for growth stories.

#### Worked example: VinFast at 1x EV/Revenue

VinFast's 2024 estimated revenue is about 1.5 billion US dollars. We apply a base-case multiple of 1x revenue, in the middle of the EV-maker comparable range.

Step 1 — Enterprise Value: `EV = revenue × multiple = \$1.5B × 1.0 = \$1.5B`. That looks far too low versus VinFast's actual listed market value, which is the first clue that the *market* is paying for a story far beyond current sales.

Step 2 — use the observable market value instead, exactly as we did for Vinhomes. VinFast's market cap in 2024 hovered around 10 billion US dollars (down from a 90-billion-dollar peak in August 2023). At a 10-billion-dollar market cap, the *implied* EV/Revenue multiple is roughly `\$10B / \$1.5B ≈ 6.7x` — wildly above the 1x our base case suggests, which tells you the listed price embeds extreme growth optimism.

Step 3 — take Vingroup's stake. VIC owns about 73% of VinFast: `0.73 × \$10B = \$7.3B`. Convert to dong: `\$7.3B × 25,000 = 182,500B VND`.

So at VinFast's 2024 market value, **182,500 billion VND** of Vingroup's value sits in this one division — more than Vinhomes, the actually-profitable business. The intuition, and the warning: a single pre-profit asset, priced on a story rather than on cash, can dominate the entire conglomerate's valuation, which makes the whole SOTP hostage to one volatile multiple.

That last point deserves emphasis. Because we are forced to anchor VinFast to its market price — and that price moved by a factor of nine in twelve months — Vingroup's SOTP is far more sensitive to VinFast's sentiment than to its own operating subsidiaries. We will quantify that sensitivity at the end.

### Why VinFast's listed price is not a clean anchor

When VinFast listed on NASDAQ in August 2023, almost none of its shares actually traded freely. Vingroup and affiliated entities held the overwhelming majority; the **free float** — the portion of shares available for the public to buy and sell — was tiny, perhaps a couple of percent. A thin float is dangerous for valuation because a small amount of buying can move the price enormously. With so few shares changing hands, a wave of enthusiasm pushed VinFast's implied market cap above 90 billion dollars, a number that valued the company above Ford and GM despite delivering a fraction of their vehicles. That price was not a sober assessment of VinFast's worth; it was the arithmetic of scarcity meeting excitement.

This is why a listed price, normally the best anchor, must be treated with suspicion for VinFast. A market cap is only as trustworthy as the depth of trading behind it. The right discipline is to cross-check the listed price against fundamentals: what EV/Revenue multiple does it imply, and is that multiple sane versus genuine comparables?

The genuine comparables are other pre-profit or thin-margin EV makers. NIO, the Chinese EV maker, traded across roughly 1x to 3x revenue through its growth years and compressed toward 1x as losses persisted. Li Auto, which reached profitability faster, earned a richer multiple. Rivian and Lucid, both US pre-profit makers, swung from over 10x revenue at peak euphoria down to under 2x as reality set in. The honest read of this comparable set is that a *durable* EV/Revenue multiple for an unproven, loss-making EV maker is somewhere around 0.5x to 2x — and that any multiple far above that is paying for a story, not for cash. VinFast's 2024 market cap of 10 billion dollars implied roughly 6.7x revenue, well outside the defensible band, which is the clearest possible signal that the listed price embeds extreme optimism.

#### Worked example: VinFast implied multiple sanity check

It is worth computing the implied multiple explicitly, because the discipline of doing so catches euphoria fast. At the August 2023 peak, VinFast's implied market cap was about 90 billion dollars on roughly 1 billion dollars of trailing revenue. Enterprise value was therefore in the ballpark of 90 billion dollars (the company had limited net cash), so the implied `EV/Revenue = 90B / 1B = 90x` (in US dollars). Ninety times revenue, for a loss-making carmaker, against peers at 0.5x to 3x. There is no fundamental story that supports a 90x revenue multiple for a capital-intensive manufacturer; the gap of roughly 30 to 180 times versus peers is pure float-driven distortion. The takeaway: always convert a market cap into an implied multiple and compare it to comparables — when the implied multiple is an order of magnitude above peers, the price is telling you about supply and demand for the stock, not about the value of the business.

![Indexed price performance of VIC, VHM, and VRE from 2021 to 2024](/imgs/blogs/sotp-valuation-vietnam-vingroup-case-study-4.png)

The price chart above shows something telling: the parent VIC fell *further* than its own subsidiaries Vinhomes and Vincom Retail over 2021-2024. If the parent were simply a transparent bundle of its subsidiaries, it should have tracked the weighted average of them. That it underperformed both is the conglomerate discount widening before your eyes.

## Valuing Vincom Retail: the cap-rate / NOI approach

Vincom Retail (ticker VRE) owns and operates shopping malls. A mall is, financially, much closer to a piece of rental property than to a manufacturer. It collects rent from tenants, pays operating costs, and the rent stream is what investors are really buying. The right lens here is the property investor's lens, not the industrialist's.

The key property metric is **Net Operating Income (NOI)**: rental revenue minus the direct operating costs of running the property (security, maintenance, utilities, property management), before financing and corporate overhead. The **cap rate** (capitalization rate) is the annual NOI divided by the property's value — effectively the rental yield a buyer demands. Rearranged, `Property value = NOI / cap rate`. A lower cap rate means buyers accept a lower yield, which means they are willing to pay *more* for the same rent — so low cap rates imply high valuations.

This is the same machinery a REIT (real estate investment trust) investor uses, and it is the correct tool because a mall's value rises and falls with its rent roll and the yield buyers demand, not with any earnings multiple borrowed from manufacturing.

#### Worked example: Vincom Retail by cap rate

Suppose Vincom Retail's portfolio generates NOI of about 4,500 billion VND a year, and prime Vietnamese retail property trades around a 10% cap rate (emerging-market retail yields run higher than developed-market ones because the perceived risk is greater).

Step 1 — capitalize the income: `Property/enterprise value = NOI / cap rate = 4,500B / 0.10 = 45,000B VND`. This lands right on Vincom Retail's observable market cap of about 45,000B VND — a satisfying cross-check that the cap-rate method and the market agree.

Step 2 — take Vingroup's stake. VIC owns about 52% of Vincom Retail: `0.52 × 45,000B = 23,400B VND`.

So **23,400 billion VND** of Vingroup's value sits in the retail malls. The intuition: when a division behaves like a landlord, value it like real estate — capitalize the rent — and ignore the temptation to slap an industrial earnings multiple on it.

Notice how each of the three listed subsidiaries used a *different* method, and each method independently roughly reconciled with the stock's market price. That reconciliation is the quiet proof that you chose the right tool for each business.

## Valuing the unlisted divisions

Some Vingroup businesses are not publicly traded, so there is no market price to anchor to. These include VinMec (private hospitals), VinSchool and VinUniversity (education), and VinBus (electric buses and logistics). For unlisted units you fall back to fundamentals: a revenue multiple drawn from comparable listed companies, or a discounted-cash-flow model if the unit produces predictable cash. The mechanics of DCF are covered in depth in the [DCF complete guide](/blog/trading/equity-research/discounted-cash-flow-dcf-complete-guide); here we use simple comparable-based estimates.

Reasonable 2024 estimates, attributable to VIC's majority stakes, are roughly: VinMec ~20,000B VND, VinSchool/VinUni ~10,000B VND, and VinBus/logistics ~5,000B VND. Together the unlisted divisions add about **35,000 billion VND**. These are the softest numbers in the whole exercise — there is no market price disciplining them — so a careful analyst would stress-test them and treat them as a smaller, more uncertain contribution rather than a precise figure.

A practical rule: the less observable a division's value, the more conservative you should be, and the more you should disclose the assumption. Padding the unlisted bucket is the easiest way to manufacture a flattering SOTP, and the easiest way to fool yourself.

#### Worked example: valuing VinMec by revenue multiple

To show the unlisted machinery concretely, take VinMec, the private hospital chain. Suppose VinMec generates about 6,000 billion VND of annual revenue. Listed private-hospital operators in Asia trade in a band of roughly 2x to 4x revenue, depending on margins and growth. Pick a conservative 3x for an unlisted, less-liquid asset that deserves a haircut for the lack of a public market: `enterprise value = 6,000B × 3 = 18,000B VND`. Net of a small amount of subsidiary debt, equity value lands near 20,000B VND, and since VIC owns a controlling majority, essentially all of that flows up. The intuition: an unlisted division is valued exactly like a listed one — pick a comparable multiple, apply it to the right metric — except you must then *discount further* for illiquidity and the absence of a market price disciplining your guess, because there is no exchange quote to catch you if you are wrong.

The same logic, applied more cautiously, gives the education and logistics units their smaller figures. Education businesses (VinSchool, VinUniversity) are valued on enrollment-driven revenue but carry thin margins and heavy reinvestment, so they earn modest multiples; the bus and logistics arm is early and capital-hungry, valued more on assets than earnings. In every case the discipline is identical: name the metric, find the comparable multiple, apply it, then haircut for illiquidity, and never let the unverifiable buckets quietly carry the valuation. A good test is to ask what happens to your total if you cut every unlisted estimate in half: if the answer barely moves the per-share number, you are safe; if it swings the thesis, you are leaning on guesses and should say so plainly rather than presenting a false precision the underlying data cannot support.

## Holding-company debt and the bridge to gross SOTP

We now have the four buckets of asset value attributable to VIC. Before we can call it a valuation, we must subtract what the *parent itself* owes. Vingroup's holding company carries net debt — borrowings at the parent level used to fund expansion, especially VinFast's capital appetite — estimated around 80,000 billion VND. This is real money that lenders are owed ahead of shareholders, so it comes straight off the top.

![Vingroup sum-of-parts waterfall chart showing each division and adjustments](/imgs/blogs/sotp-valuation-vietnam-vingroup-case-study-2.png)

The waterfall above is the whole calculation in one picture: the green and amber bars are the division stakes building value up, the red bar is the parent's net debt pulling it down, the blue bar is the gross sum, and the final lavender bar is what survives after the conglomerate discount we are about to apply. The dashed line is the actual market cap — and the distance between the blue gross-SOTP bar and that line is, once again, the discount.

#### Worked example: full SOTP table with a 20% conglomerate discount

Let us assemble everything into the bridge from segment values to an implied per-share value.

Sum of stakes attributable to VIC:

- Vinhomes (69%): 138,000B VND
- VinFast (73%, at \$10B market cap): 182,500B VND
- Vincom Retail (52%): 23,400B VND
- Unlisted divisions: 35,000B VND
- **Subtotal of asset stakes: 378,900B VND**

Subtract holding-company net debt: `378,900B − 80,000B = 298,900B VND`. This is the **gross SOTP equity value** — what Vingroup would be worth if the market gave it full credit for its parts.

Now apply the conglomerate discount. We use 20%, squarely in the typical emerging-market range:

`Discounted SOTP = 298,900B × (1 − 0.20) = 298,900B × 0.80 = 239,120B VND`.

Compare to reality. Vingroup's actual market cap is about 180,000B VND. So the market is applying an *even deeper* discount than our 20%: the implied discount is `1 − (180,000 / 298,900) ≈ 40%`. Either the market is too pessimistic and VIC is undervalued, or the market is right and our gross parts (especially the story-driven VinFast slice) are too generous.

Step to per-share: if Vingroup has roughly 3.8 billion shares outstanding, our 20%-discounted SOTP of 239,120B VND implies about `239,120B / 3.8B ≈ 62,900 VND per share`, versus a market price implied by the 180,000B cap of roughly `180,000B / 3.8B ≈ 47,400 VND per share`.

The intuition: our parts-based fair value sits meaningfully above the market price, which frames the entire investment debate as a single question — is the extra discount the market demands justified, or is it an opportunity?

That question turns almost entirely on two things: whether you believe the VinFast slice, and whether the conglomerate discount should be 20% or 40%. We tackle both next.

## Why the conglomerate discount exists in Vietnam

The conglomerate discount is not a fudge factor; it is the market pricing specific, nameable risks. In Vietnam, and emerging markets generally, several of these are pronounced.

**Family and founder control.** Vingroup is closely associated with its founder. Concentrated control means outside minority shareholders have limited say. Decisions that benefit the controlling shareholder's broader interests — but not necessarily the minority's — are harder to block. Markets demand a discount for that loss of control.

**Related-party transactions.** When a parent and its subsidiaries trade with one another — the parent sells land to the property arm, or the property arm buys vehicles from the car arm — the prices may not be arm's-length. Value can quietly migrate from a division you own a lot of to one you own less of. Outside investors cannot easily audit these flows, so they discount for the uncertainty.

**Cross-guarantees and pledged subsidiaries.** This is the sharpest risk. To fund VinFast's enormous capital needs, the group has at times pledged shares in its healthy, cash-generating subsidiaries as collateral, or guaranteed parent debt with subsidiary strength. That means a problem at the cash-burning division can reach back and threaten the value of the profitable ones. The divisions are not as independent as the SOTP arithmetic pretends.

**Information opacity.** A conglomerate's consolidated financial statements blend everything together. It is genuinely hard for an outside analyst to see each division's true economics. Less visibility means more uncertainty, and uncertainty is priced as a discount.

**Capital trapped in the structure.** Cash generated by Vinhomes might be the most valuable thing in the group, but if it is being redirected to fund VinFast's factories, a Vinhomes-only investor never sees it. Owning the parent means accepting that your claim on the best cash flows is diluted by the group's other priorities.

**Currency and country risk on top.** Because part of the value (VinFast) is denominated in US dollars on NASDAQ while the parent and most subsidiaries are priced in dong on the Ho Chi Minh exchange, a SOTP must convert across currencies, and the exchange rate itself becomes a risk factor. A weakening dong raises the dong value of the dollar-denominated VinFast stake mechanically, even if nothing about VinFast's business changes — a reminder that a cross-listed conglomerate carries currency mismatch that a single-country company does not. Emerging-market investors also demand a higher required return generally, which compresses every multiple in the stack relative to a developed-market peer. None of these frictions is hypothetical; each one is a line a careful analyst can point to and quantify, which is why the discount is a sober estimate rather than a mood.

![Bar chart comparing Vingroup SOTP estimate to actual market cap from 2020 to 2024](/imgs/blogs/sotp-valuation-vietnam-vingroup-case-study-5.png)

The chart above tracks the gap between the sum-of-parts estimate and the actual market cap from 2020 to 2024. The discount is not constant — it widened as VinFast's losses mounted and the group's debt load grew, which is exactly what you would expect: the discount is the market's running tally of how much it distrusts the structure, and that distrust grew.

### How big should the discount be? Calibrating, not guessing

The discount is the most subjective input in SOTP, but subjective does not mean arbitrary. You calibrate it against observable evidence rather than picking a round number. There are three practical anchors.

The first anchor is **the historical discount the same company has traded at.** If Vingroup has spent the last three years between a 30% and 45% discount to its own sum-of-parts, then assuming a 10% discount today requires a specific reason — some catalyst that will narrow it. Absent a catalyst, the base case is roughly where it has been. The 2020-2024 chart gives exactly this anchor: the implied discount sat in the high-30s to 40s, which is why our 20% assumption produced a fair value *above* market — we were implicitly betting the discount would narrow.

The second anchor is **peer conglomerates.** Korean chaebols, Indian family groups, and Latin American holding companies trade at discounts that cluster by governance quality. Groups with strong minority protections and clean related-party records trade at smaller discounts (10-15%); groups with concentrated control, opaque intragroup dealings, and cross-guarantees trade wider (25-40%). Placing Vingroup against that spectrum — concentrated founder control, active intragroup funding of VinFast, pledged subsidiary shares — argues for the wider end.

The third anchor is **the catalyst question: what would close the discount?** A discount narrows when the group simplifies — spins off a division, lists a subsidiary so its value becomes observable, or commits to a capital-return policy. It widens when the group does the opposite: funds a cash-burning bet with the cash flows of a healthy subsidiary. Vingroup spent 2021-2024 doing the latter, which is why the discount widened rather than narrowed. A disciplined analyst sets the discount not by feel but by asking, concretely, which of these forces is currently dominant.

The honest conclusion for Vingroup is uncomfortable: a 20% discount is the *optimistic* case (it assumes the gap narrows), and the market's revealed ~40% is the realistic base case given current governance and capital allocation. Stating that explicitly — rather than burying it in a single point estimate — is the difference between a valuation and a guess.

## Building the model in practice: a checklist

Translating the method into a spreadsheet is mechanical once you respect the order of operations. The sequence that prevents the common errors:

1. **List every division and the parent's exact stake.** Pull stakes from the latest annual report, not memory; they change as the group buys, sells, or dilutes.
2. **For each listed subsidiary, record the live market cap** and compute the implied multiple, then sanity-check that multiple against industry comparables. If the implied multiple is wild (VinFast), flag it and decide whether to anchor to the market price or to a fundamental multiple — and *state which you chose*.
3. **For each unlisted subsidiary, build a small DCF or apply a comparable multiple to its revenue or EBITDA.** Be conservative; these are unverifiable.
4. **Multiply each subsidiary's equity value by the parent's stake.** This is the step everyone forgets. Write the stake percentage in its own column so the error is impossible to hide.
5. **Sum the stake values, then subtract the parent's net debt and any unallocated holding-company costs.** Use the parent-only ("standalone") balance sheet for this debt, not the consolidated one, to avoid double-counting subsidiary debt already reflected in subsidiary equity values.
6. **Apply a calibrated conglomerate discount,** justified against the company's own history and peer governance.
7. **Divide by shares outstanding** for a per-share value, and present a *range* driven by the two or three inputs that matter most — here, the VinFast multiple and the discount.

#### Worked example: turning the SOTP into a per-share range

A single point estimate is false precision; a range is honest. Take the two swing inputs. On the discount, span 20% (optimistic, gap narrows) to 40% (current market reality). On VinFast, span anchoring to its market price (generous) versus a fundamental 1x revenue (cautious).

Optimistic corner — VinFast at market price, 20% discount: gross SOTP 298,900B × 0.80 = 239,120B; per share `239,120B / 3.8B ≈ 62,900 VND`.

Pessimistic corner — VinFast at fundamental 1x revenue (contribution ~27,400B instead of 182,500B, lowering gross SOTP to 298,900B − 182,500B + 27,400B = 143,800B), then a 40% discount: `143,800B × 0.60 = 86,280B`; per share `86,280B / 3.8B ≈ 22,700 VND`.

So a defensible fair-value *range* runs from roughly 22,700 to 62,900 VND per share, bracketing the market's implied ~47,400 VND. The intuition: the market price sits inside our range, near the middle, which means Vingroup is neither an obvious bargain nor an obvious short — the answer depends almost entirely on whether you believe VinFast's story, and a range makes that dependence honest instead of hiding it behind one tidy number.

## When to use SOTP — and when not to

SOTP is the right tool when a company is genuinely several businesses that should be valued differently. The clearest triggers:

- **Divergent business models.** A profitable developer plus a pre-profit carmaker plus a landlord cannot share one multiple. Vingroup is the textbook case.
- **A pre-profit or loss-making division.** The moment one piece has no earnings to multiply, a blended P/E is impossible and SOTP becomes mandatory.
- **Listed subsidiaries.** When divisions are separately traded (Vinhomes, VinFast, Vincom Retail), you get live market prices for the parts — a gift that makes SOTP both easier and more credible.
- **A suspected conglomerate discount.** If you think the market is mispricing the whole versus the parts, SOTP is the only way to measure the gap and decide whether it is an opportunity.

SOTP is the *wrong* tool, or at least overkill, when a company is one coherent business. Applying SOTP to a focused single-product company invents divisions that do not exist and adds false precision. Use a single appropriate method — DCF or a peer multiple — for those. For how a whole-market lens differs from a single-company one, see [Vietnam stock market valuation](/blog/trading/asset-valuation/vietnam-stock-market-valuation-vnindex-pe-dynamics).

The most common SOTP mistakes, all visible in the Vingroup case:

1. **Counting 100% of a subsidiary you only partly own.** VinFast is worth ~\$10B, but only 73% of that belongs to VIC. Forgetting the stake overstates value by billions.
2. **Forgetting holding-company debt.** Skipping the 80,000B parent net debt makes Vingroup look 80,000B richer than it is.
3. **Double-counting.** If a subsidiary's value already reflects an asset, do not also count that asset separately in the parent.
4. **Ignoring the discount.** A gross SOTP that the market will never pay is a fantasy number; the discount is what makes it actionable.
5. **Padding unlisted divisions.** The unverifiable buckets are where wishful thinking hides.

## Sensitivity: how much does VinFast move the answer?

Because VinFast is valued on a story-driven multiple rather than on cash, and because that multiple has proven wildly unstable, the single most important sensitivity in the whole valuation is the EV/Revenue multiple we assign to VinFast. Everything else — Vinhomes, Vincom Retail, the unlisted units, the parent debt — we can hold fixed and watch what VinFast alone does to the answer.

#### Worked example: VinFast multiple sensitivity

Hold the rest of the SOTP constant. The non-VinFast pieces are: Vinhomes 138,000B + Vincom Retail 23,400B + unlisted 35,000B − parent debt 80,000B = **116,400B VND**, fixed. VinFast's contribution is `multiple × \$1.5B revenue × 73% stake × 25,000 VND/USD`.

- **Bear case, 0.5x revenue:** VinFast = `0.5 × \$1.5B × 0.73 × 25,000 = ~13,700B VND`. Gross SOTP = `116,400B + 13,700B ≈ 130,100B VND` — *below* the current 180,000B market cap. At this multiple, the market is already too optimistic.
- **Base case, 1.0x revenue:** VinFast = `~27,400B VND`. Gross SOTP = `116,400B + 27,400B ≈ 143,800B VND` — still below market cap.
- **Bull case, 2.0x revenue:** VinFast = `~54,800B VND`. Gross SOTP = `116,400B + 54,800B ≈ 171,200B VND` — finally approaching the market cap.

Notice this is a *different* lens than our earlier full SOTP, where we anchored VinFast to its observed \$10B market cap (an implied ~6.7x revenue). Here we instead value VinFast on fundamentals using EV/Revenue multiples of comparable carmakers — and at every reasonable fundamental multiple (0.5x to 2x), the gross SOTP comes out *below* Vingroup's market cap. The takeaway is sobering: if you do not believe VinFast deserves a multiple far above its automotive peers, then on fundamentals Vingroup is not obviously cheap at all — and the bullish SOTP case rests entirely on accepting VinFast's lofty market price.

![Bar chart of VinFast EV/Revenue scenarios and their impact on Vingroup gross SOTP value](/imgs/blogs/sotp-valuation-vietnam-vingroup-case-study-7.png)

The sensitivity chart makes the dependence stark: VinFast's multiple is the dial that swings the whole valuation across the market-cap line. This is the deepest lesson of the Vingroup case. A sum-of-parts is only as solid as its softest, most volatile part — and when one story-driven division dominates, the SOTP inherits all of that story's fragility.

## Common misconceptions

**"The whole should always equal the sum of the parts."** Arithmetically yes, but in markets, almost never for conglomerates. Vingroup's gross parts sum to ~298,900B VND while it trades near 180,000B — a ~40% implied discount. The gap is real and persistent, driven by control, opacity, and cross-guarantee risks, not by a calculation error.

**"A listed subsidiary's market cap is the truth, so just use it."** It is the best *available* anchor, but it can be a story price, not a fundamental one. VinFast at a \$10B market cap implies ~6.7x revenue versus peers at 0.5-2x. Anchoring to the market price imports the market's optimism wholesale; valuing on fundamentals tells a much more cautious tale, as the sensitivity showed.

**"SOTP gives you the share price target."** SOTP gives you a *gross* parts value; the market price reflects that minus a conglomerate discount. Our gross 298,900B becomes 239,120B after a 20% discount, and the market applies ~40%. Skipping the discount overvalues the company by tens of percent.

**"You can value the whole conglomerate on one P/E."** Impossible when a division has no earnings. VinFast's losses make consolidated earnings a distorted, often meaningless base. Any single multiple applied to Vingroup is arithmetic theater.

**"A bigger conglomerate is safer because it is diversified."** Diversification across unrelated businesses sounds like it should reduce risk, but for outside shareholders it often does the opposite. A focused company can be analyzed cleanly and its cash returned to you; a diversified group adds opacity, intragroup transfers, and the risk that a healthy business subsidizes a failing one. The market knows this, which is precisely why diversified conglomerates trade at a *discount* rather than a premium. Vingroup's breadth is not a comfort to its minority shareholders; it is a source of the discount, because each added business is one more place value can leak before it reaches them.

**"Owning the parent is the same as owning the good subsidiary."** It is not. Owning VIC for its Vinhomes exposure means accepting that Vinhomes' cash may fund VinFast, that Vinhomes shares may be pledged for parent debt, and that you only get 69% of Vinhomes diluted by everything else. If you want Vinhomes, many investors simply buy VHM directly — which is itself part of why the parent trades at a discount.

## How it shows up in real markets

The Vingroup discount is a live, tradeable phenomenon, and it has a clean real-world test. Throughout 2023-2024, Vietnamese and foreign analysts repeatedly published SOTP estimates for VIC well above its market price, while the stock kept drifting lower (the indexed chart showed VIC falling to ~40 versus VHM ~50 and VRE ~55 by end-2024). The parts looked cheap; the whole stayed cheap. That is the discount refusing to close — and it refused because the underlying risks (parent debt funding VinFast, pledged subsidiary shares) kept getting *more* acute, not less.

The August 2023 VinFast listing is the other vivid lesson. For a few days, VinFast's NASDAQ market cap implied that VIC's 73% stake alone was worth more than five times Vingroup's entire market value — an obvious impossibility that pure arithmetic flagged instantly. SOTP discipline told you immediately that the VinFast price was a thin-float story price, not a number to anchor a parent valuation on. Analysts who plugged the peak VinFast price into a Vingroup SOTP produced absurd fair values; those who treated it with appropriate skepticism, or used fundamental EV/Revenue multiples, stayed sane.

The general pattern repeats across emerging-market conglomerates worldwide: Korean chaebols, Indian family groups, Latin American holding companies all tend to trade below their sum-of-parts, and the discount widens when the controlling family pursues expensive, controversial bets funded by the group's healthier cash flows. The discipline is always the same — value each part on its own terms, subtract the parent's debt, take only the stake you own, and then ask honestly how big the discount should be given who controls the group and how the cash actually moves. For how required returns and discount rates feed into the per-division valuations, the relationship between [interest rates, bonds, and stocks](/blog/trading/macro-trading/interest-rates-bonds-stocks-relationship) is the backdrop against which all these multiples expand and compress.

### The investor's escape hatch: buy the part, not the parent

SOTP does not just produce a number; it produces a *decision*. Once you have valued each division and seen the discount, you face a practical choice that the parent's own shareholders quietly make all the time: if you want exposure to one of Vingroup's businesses, you can often buy that business directly. Want Vinhomes? Buy VHM on the Ho Chi Minh exchange and get 100% pure exposure with no parent debt, no VinFast losses, no conglomerate discount. Want the EV story? Buy VFS on NASDAQ directly. The only reason to buy the parent VIC instead is if you believe the conglomerate discount is *too wide* — that the market is over-punishing the structure and will eventually narrow the gap.

This is the deepest practical payoff of doing the SOTP. It reframes the question from "is Vingroup cheap?" to "is the discount on Vingroup wider than it deserves to be, and is there a catalyst to close it?" If the answer is no — if the discount is fair given the governance and the cash-allocation risks — then a Vinhomes bull should simply buy Vinhomes. The existence of listed subsidiaries turns the conglomerate discount from a vague worry into a measurable arbitrage that anyone can act on, and it is one big reason the discount persists: rational investors who only want one division route around the parent, leaving the parent permanently under-bid.

### What would actually close the Vingroup discount

It helps to name the concrete catalysts, because a SOTP is most actionable when paired with a view on whether the gap will narrow. The discount on Vingroup would compress if the group did any of the following: listed or spun off VinFast cleanly so its value became transparently observable and the cross-funding stopped; committed to a dividend or buyback that returned subsidiary cash to parent shareholders rather than recycling it into capital projects; unwound the share pledges that tie healthy subsidiaries to parent debt; or simply reached the point where VinFast turned cash-flow positive and stopped being a drain on the group. Each of those is a real, watchable event. Conversely, the discount would widen further on any new large capital commitment funded by subsidiary strength, any sign of related-party transactions moving value toward the controlling shareholder, or any deterioration in the group's debt position. The SOTP gives you the gap; the catalyst list tells you which way it is likely to move and what to watch for. Pairing the two is what separates a static valuation from an investable thesis.

The Vingroup case, in one line: a sum-of-parts is the only honest way to value a conglomerate, but it is only as reliable as its softest part — and when a single story-driven division dominates the sum, the entire valuation lives and dies on whether you believe that story.

## Further reading & cross-links

- [Sum-of-parts valuation for conglomerates and divisions](/blog/trading/asset-valuation/sum-of-parts-valuation-sotp-conglomerates-divisions) — the general SOTP method this case study applies.
- [Real estate valuation: cap rate, NOI, and DCF](/blog/trading/asset-valuation/real-estate-valuation-cap-rate-noi-dcf) — the property method behind the Vincom Retail and Vinhomes pieces.
- [Vietnam stock market valuation: VN-Index P/E dynamics](/blog/trading/asset-valuation/vietnam-stock-market-valuation-vnindex-pe-dynamics) — the market backdrop for valuing Vietnamese companies.
- [Enterprise value vs market cap and implied growth](/blog/trading/asset-valuation/enterprise-value-vs-market-cap-implied-growth-rates) — the EV-to-equity bridge used in every division above.
- [Sum-of-parts valuation (equity research view)](/blog/trading/equity-research/sum-of-parts-valuation) — the analyst-workflow companion to this method.
- [Discounted cash flow: a complete guide](/blog/trading/equity-research/discounted-cash-flow-dcf-complete-guide) — the DCF used for unlisted divisions.
