---
title: "Transfer Taxes and Capital Gains on Property: What the State Takes When You Sell"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "A beginner-friendly deep dive into how the state taxes a property sale: Vietnam's flat 2% tax on the whole price versus the US capital-gains model, the primary-residence exclusion, the 1031 exchange, step-up in basis at death, and the lock-in effect that freezes sellers in place."
tags: ["real-estate", "property", "capital-gains-tax", "transfer-tax", "1031-exchange", "step-up-in-basis", "lock-in-effect", "vietnam", "land-law-2024", "personal-finance"]
category: "trading"
subcategory: "Real Estate"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — When property changes hands, the state takes a cut, and the *design* of that cut quietly shapes the whole market. Vietnam and the United States tax the same sale on two completely different bases — and that single design choice explains a lot of how each market behaves.
>
> - **Vietnam** charges a flat **2% personal-income tax on the transfer *price*** (not the profit) plus a **0.5% registration fee**. It is dead simple — but crude (you pay even when you sell at a loss) and gameable (buyers and sellers under-declare the contract price to shrink the tax).
> - **The United States** taxes the actual **capital gain** — the profit — at long-term rates of **0%, 15%, or 20%**, but softens it heavily: a **primary-residence exclusion** of **\$250,000 single / \$500,000 married**, a **1031 like-kind exchange** that lets investors defer the gain indefinitely, and a **step-up in basis at death** that can erase it entirely.
> - The two designs push behaviour in opposite directions: Vietnam's price-based tax discourages *trading* (and invites under-declaration); the US's gain-based tax plus its exclusions discourage *selling* (the "lock-in effect" that freezes owners in place).
> - The one number to remember: on a ₫7.0 billion (≈ \$270,000) HCMC sale, Vietnam's transfer taxes come to about **₫175 million** — owed even if you lost money. On a US home with a \$220,000 gain, a married couple's exclusion can drop the tax to **\$0**.

In the same week in 2025, two people sold an apartment.

In Ho Chi Minh City, a man we'll call **Minh** sold the ₫7.0 billion (≈ \$270,000) flat he'd bought four years earlier. He'd actually done well — the market had run hard — but when his notary handed him the tax slip, the number didn't depend on his profit at all. It was simply 2% of the sale price: ₫140 million in personal-income tax, plus a ₫35 million registration fee for the buyer. ₫175 million, gone, calculated off the *headline price* as if profit were irrelevant. His agent leaned in and quietly suggested what half the market does — *put a lower number on the contract*. Minh hesitated.

In suburban Ohio, a married couple we'll follow as **Dana** and her husband sold the home they'd lived in for nine years for \$520,000. They'd bought it for \$300,000 and put \$20,000 into a new kitchen, so their profit — their *gain* — was around \$170,000 after the costs of selling. In Vietnam that would have triggered a tax on the whole \$520,000. In the US, Dana owed the federal government exactly **\$0** — because a married couple selling their main home can exclude up to \$500,000 of gain, and \$170,000 fits comfortably underneath. She didn't even have to report it.

Same transaction — a home changing hands — two completely different tax bills, built on two completely different ideas of *what* should be taxed. That difference is the whole subject of this article, and it is far more consequential than it looks. The way a country taxes a property sale doesn't just determine who pays what; it quietly shapes who sells, who holds, who trades, and even whether people tell the truth on the contract.

![A two-column comparison showing Vietnam taxing the whole sale price at a flat rate while the United States taxes only the capital gain and shelters most of it with an exclusion](/imgs/blogs/transfer-taxes-and-capital-gains-on-property-1.png)

The figure above is the mental model for the entire post. On the left, Vietnam: the **base** of the tax is the *whole sale price*, the rate is a flat 2%, and the bill is owed even at a loss. On the right, the United States: the base is only the *gain* — the profit — taxed at a graduated rate, and then most of that gain is sheltered by an exclusion. Two bases, two philosophies. By the end you'll understand both systems from the ground up, be able to compute the tax on any sale yourself, and — more interesting — see how each design bends the market around it. We'll keep Minh (in HCMC, in ₫) and Dana (in the US, in \$) as our running examples throughout, so a Vietnamese reader and an international one both stay oriented.

A quick honesty note before we begin: this is educational, not individualized tax advice. Tax codes are intricate and change; the goal here is to hand you the *mechanism* — the logic behind the rules — so you understand what's happening when you sell, not to tell you what to do. Always check the current law (and a professional) before you transact.

## Foundations: transfer tax versus capital-gains tax

Let's build the whole thing from zero, because almost every confusion about property tax-on-sale comes from blurring two completely different ideas: taxing the *price* and taxing the *profit*. They sound similar. They are not. Getting the distinction crisp is 80% of understanding the rest.

When you sell a property, a tax authority can take its cut in one of two fundamentally different ways. It can tax a slice of the **transaction value** — the headline price the property sold for — regardless of whether you made or lost money. Or it can tax a slice of your **gain** — the profit, meaning how much more you sold for than you originally paid. These are the two camps, and most countries lean to one side or the other. Vietnam is firmly in the first camp; the United States is firmly in the second.

### Transfer tax: a tax on the price

A **transfer tax** is a levy on the *act* of transferring ownership — a percentage of the price, charged simply because the property changed hands. It doesn't care about your profit. It's a tax on the *transaction*, not on the *outcome*. You could buy at ₫7 billion, sell at ₫6 billion (a real loss), and still owe a transfer tax on the ₫6 billion sale, because the tax attaches to the event of selling, not to whether you came out ahead.

Many places have some flavour of this. In the UK it's "stamp duty"; in much of the US, a "real-estate transfer tax" or "deed-recording tax" levied by states and cities (often modest — a fraction of a percent). In Vietnam, the seller's headline tax on a property sale — a 2% **personal income tax** — is computed off the transfer price, which makes it function exactly like a transfer tax even though it carries the name of an income tax. We'll dig into why that label matters later, but for intuition: Vietnam's "PIT on transfers" *behaves* like a transfer tax, because its base is the price, not the gain.

### Registration fee: the cost of recording ownership

Closely related is the **registration fee** — in Vietnamese, *lệ phí trước bạ*. This is the charge for officially registering the new owner in the state's land and property records. Think of it as the state's filing fee for stamping "this is now yours" into the public ledger that protects your ownership. In Vietnam it's **0.5%** of the property's value. It's usually the buyer who pays it (whereas the 2% income tax is the seller's, by default — though in practice the two sides negotiate who actually bears each). The registration fee, like the transfer tax, is computed on *value*, not profit.

### Capital gain: a tax on the profit

The other camp taxes the **capital gain**. To understand it we need three sub-definitions, built carefully because the whole US system hangs on them.

A **capital asset** is something you own as an investment or for personal use that can rise or fall in value — a house, a share of stock, a plot of land, a painting. A **capital gain** is the profit you make when you sell that asset for more than it cost you. If you bought a house for \$300,000 and sell it for \$500,000, your capital gain is \$200,000. If you sell for \$250,000, you have a **capital loss** of \$50,000. The tax authority in the "gain" camp taxes the *gain*, not the price.

To compute the gain you need the **cost basis** — often just called the *basis*. The cost basis is essentially *what the asset cost you*: the original purchase price, plus certain costs you put in along the way. For a house, the basis is the purchase price plus the cost of capital improvements (a new roof, an addition, a renovated kitchen — things that add lasting value), plus some buying costs. It is *not* increased by ordinary repairs or by the mortgage interest you paid. The basis is your "starting line" — the number you subtract from the sale price to find the profit.

$$\text{capital gain} = \text{sale price} - \text{cost basis} - \text{selling costs}$$

where *selling costs* are the agent's commission, legal fees, and closing costs you pay to sell. Every symbol here is in money; the gain is what's left over and the thing that gets taxed in the US model.

### Realized versus unrealized gain

One more crucial distinction. Your gain can be **unrealized** or **realized**, and only one of them is taxable.

An **unrealized gain** is a profit on paper. If your house has risen in value from \$300,000 to \$500,000 but you haven't sold it, you have an unrealized gain of \$200,000 — you're \$200,000 "richer" in principle, but you haven't turned it into cash and, crucially, you owe no tax on it. You can sit on an unrealized gain for forty years and never pay a cent. A **realized gain** is what happens the moment you *sell*: the paper profit becomes a real, cash profit, and *that* is the taxable event. The single most important sentence in the entire US capital-gains system is this: **gains are taxed only when realized — only when you sell.** Hold, and you defer the tax forever. Remember that sentence; half of this article (the exclusions, the 1031 exchange, the step-up, the lock-in effect) is just consequences of it.

That sentence is *also* why the two systems behave so differently. In Vietnam, the tax attaches to the *transaction* — so the disincentive is to *transact*, to trade, to flip. In the US, the tax attaches to the *realized gain* — so the disincentive is to *realize*, to sell at all. Same money, very different behaviour, as we'll see.

#### Worked example: the same sale, two tax bases

Let's make the abstraction concrete immediately, because it's the spine of everything. Take one sale: a property bought for the equivalent of \$300,000 and sold four years later for \$500,000. The owner made a \$200,000 profit.

- **Under a price-based (transfer) tax at 2%:** the tax is `2% × \$500,000 = \$10,000`. Notice: the profit (\$200,000) never entered the calculation. If the owner had *lost* money — sold for \$280,000 — the tax would have been `2% × \$280,000 = \$5,600`, still owed, on a loss.
- **Under a gain-based (capital-gains) tax at 15%:** the tax is `15% × \$200,000 = \$30,000`. Here the profit is everything; if the owner sold at a loss, the tax would be **zero** (and the loss might even offset other gains).

*The price-based tax punishes turnover regardless of profit; the gain-based tax punishes profit regardless of turnover — and that single difference radiates through both markets.*

With those foundations — transfer tax, registration fee, cost basis, realized versus unrealized gain — we can now look at each country's system in full.

## Vietnam's flat 2% transfer tax (plus registration): simple but crude

Vietnam's system is admirably simple to state and faintly brutal in its logic. When you sell residential property as an individual, you owe a **personal income tax of 2% of the transfer price**. Full stop. There is no calculation of profit, no cost basis, no deduction for the kitchen you renovated, no allowance for inflation. The state looks at the price on the contract and takes 2%. On top of that, the buyer pays a **0.5% registration fee** (*lệ phí trước bạ*) to record the new ownership. Those two charges — the seller's 2% and the buyer's 0.5% — are the headline taxes on a Vietnamese property sale.

A small but important nuance: although it's labelled a "personal income tax", the 2% is **not** computed on income (profit) — it's computed on the gross transfer value. Vietnam *used* to give sellers a choice between 25% of the actual gain or 2% of the price, but in practice the 2%-of-price method became the standard because gains are hard to prove (basis records are spotty) and the flat method is administratively trivial. So in effect, Vietnam taxes the *price*, like a transfer tax wearing an income-tax name tag.

Two more mechanical points worth pinning down, because they trip up first-time sellers. First, **who actually pays.** By statute the 2% PIT is the *seller's* liability and the 0.5% registration fee is the *buyer's* — but in a hot market that allocation is just a starting point for negotiation. It's common for a seller to insist on a "net" price and push the 2% onto the buyer, or for the two sides to split the costs; the *law* assigns them, but the *market* reassigns them, exactly as it does with any transaction cost. Second, the tax base isn't *purely* the contract price in the way you'd assume: if the price you declare is *below* the state's official land-and-housing price table for that location, the tax authority computes the 2% on the **higher of the two** — the declared price or the state table price. That floor is precisely the lever Land Law 2024 sharpens, and we'll come back to it. Third, there is a narrow **exemption**: a transfer between close family members (spouses, parents and children, siblings) is exempt from the 2% PIT — a carve-out for genuine intra-family transfers rather than market sales. Outside that carve-out, the 2% is essentially unavoidable on an arm's-length sale.

### Why "simple" is a genuine virtue

Don't underrate the simplicity. A flat percentage of a single, observable number — the sale price — is something a notary can compute in ten seconds, a seller can predict to the dong, and a tax office can verify without forensic accounting. There's no argument about what counts as a capital improvement, no decades-old receipts to dig up, no fight over depreciation. For a fast-developing country with limited administrative capacity and millions of informal transactions, a 2% flat tax is a rational design: it's cheap to collect, hard to game on the *rate*, and predictable. Compared to the US system you're about to meet — which can require a tax professional to compute correctly — Vietnam's is refreshingly legible. That legibility is the whole appeal.

### Why "crude" is the price of that simplicity

But simplicity has a sharp edge: a price-based tax is **blind to whether you actually made money**. This is its central flaw. Consider three Vietnamese sellers, all selling for ₫7 billion:

- One bought at ₫3 billion and is sitting on a ₫4 billion profit.
- One bought at ₫7 billion and is breaking even.
- One bought at ₫9 billion (near a market peak) and is selling at a **₫2 billion loss**.

All three pay the **same** ₫140 million in tax (2% of ₫7 billion). The seller nursing a ₫2 billion loss pays exactly as much as the one with a ₫4 billion windfall. In the gain-based US system, the loss-maker would owe nothing. In Vietnam, the tax doesn't care. This is why a price-based tax is described as *regressive with respect to profit*: it bears hardest, proportionally, on the people who did worst.

There's a second crudeness: the tax doesn't adjust for **inflation** or holding period. If you held a property for fifteen years and its price tripled mostly because the currency lost value, a gain-based system *might* let you index the basis to inflation (some countries do); the 2%-of-price tax doesn't engage with any of that — it just takes 2% of the nominal price whenever you sell. And it doesn't distinguish a quick flip from a lifetime home: a speculator who buys and sells in six months and a family selling the home they raised three children in pay the identical 2%.

#### Worked example: Minh's ₫7 billion sale (and what happens if he sells at a loss)

Let's run Minh's actual numbers. He sells his HCMC flat for **₫7.0 billion (≈ \$270,000)**.

- **Personal income tax (seller):** `2% × ₫7.0 billion = ₫140 million (≈ \$5,400)`.
- **Registration fee (buyer):** `0.5% × ₫7.0 billion = ₫35 million (≈ \$1,350)`.
- **Combined transfer taxes on the deal:** `₫140M + ₫35M = ₫175 million (≈ \$6,750)`.

Minh bought at ₫5 billion, so he made a ₫2 billion gross profit — his ₫140M tax is a modest 7% of his profit, which feels fair enough. But now imagine the market had turned and Minh were selling the *same* flat for ₫7 billion having bought it at ₫8 billion near a peak — a ₫1 billion *loss*. His PIT would *still* be `2% × ₫7.0 billion = ₫140 million`. He'd be handing the state ₫140 million on a transaction where he lost ₫1 billion. There is no loss relief, no carry-forward, nothing.

*In Vietnam the question "did you profit?" is irrelevant to the tax — you pay 2% of the price whether you struck gold or got crushed, and that single fact is why the system is simple to run and unfair to the unlucky.*

This crudeness — the fact that you pay real money even on a loss, and the fact that the tax scales with the *price* — is exactly what drives the most pervasive behaviour in the Vietnamese property market: under-declaration.

## The under-declaration game, and why Land Law 2024 attacks it

Here is the open secret of Vietnamese property: a very large share of transactions are recorded on the contract at a price *well below* what actually changed hands. The seller and buyer agree on a real price — say ₫7 billion — and then write a *lower* number — say ₫4 billion — onto the official sale contract that gets filed for tax. The difference, ₫3 billion, moves in cash or by private transfer, off the books. Because the 2% PIT and the 0.5% registration fee are both computed on the *declared* price, lowballing the contract directly shrinks the tax.

**Under-declaration** is, plainly, the practice of stating a transaction value lower than the true one to reduce a value-based tax. It is the predictable response to a price-based tax: when the tax is a percentage of a *number you write down*, and that number is hard for the state to independently verify, people write down a smaller number. The incentive is baked into the design. You don't get this behaviour with a gain-based tax in the same way, because there the *cost basis* is also on record (the price the previous owner declared), so lowballing the sale just shifts the problem to the next sale — the buyer who under-declares today inherits a low basis and faces a *bigger* taxable gain when they sell. A price-based tax has no such self-correcting mechanism. Everyone benefits from a low number, every time.

#### Worked example: the under-declaration math (and the risk)

Suppose Minh and his buyer agree the real price is **₫7.0 billion** but declare **₫4.0 billion** on the contract.

- **Honest declaration (₫7.0 billion):** PIT = `2% × ₫7.0bn = ₫140M`; registration = `0.5% × ₫7.0bn = ₫35M`. Total to the state: **₫175 million**.
- **Under-declared (₫4.0 billion):** PIT = `2% × ₫4.0bn = ₫80M`; registration = `0.5% × ₫4.0bn = ₫20M`. Total to the state: **₫100 million**.
- **Tax "saved":** `₫175M − ₫100M = ₫75 million (≈ \$2,900)` — split, by negotiation, between the two parties.

That ₫75 million is real money, and you can see why the temptation is overwhelming when the practice is normalized. But the savings come with genuine, often under-appreciated risks:

- **It's illegal — tax evasion.** If the tax authority detects the gap (and they increasingly can, by comparing to neighbouring sales and the new land-price tables), the parties owe **back taxes plus penalties**, and large or repeated cases can be prosecuted criminally.
- **The buyer is badly exposed.** The *buyer's* legal record of ownership now says they paid ₫4 billion. If a dispute arises — the deal collapses, the seller reneges, there's litigation — the buyer can typically only reclaim the ₫4 billion the contract proves they paid, not the ₫7 billion they actually handed over. The ₫3 billion in cash has no legal footprint.
- **It poisons the next sale.** When the buyer later sells, their *acquisition* price on record is the lowball ₫4 billion. They've kicked the can — and if Vietnam ever moves to a true gain-based tax, that low recorded basis would inflate their future taxable gain.

![A two-column comparison of a Vietnam sale showing a low declared contract price filed for tax versus the higher real price paid off the books, and the risks attached](/imgs/blogs/transfer-taxes-and-capital-gains-on-property-6.png)

The figure above lays the game out: on the left, the official contract (₫4 billion, the tax base shrunk by 2%); on the right, the real ₫7 billion that changed hands, the tax avoided, and the legal exposure. The state loses revenue, the data on actual prices is corrupted (which makes everything from policy to mortgage underwriting harder), and the buyer trades a tax saving for a legal blind spot.

### How Land Law 2024 attacks the problem

The Vietnamese state has long known about under-declaration; the hard part is *detecting* it without a reliable independent benchmark for what a property is "really" worth. For years the benchmark was a government **land-price framework** set on a *five-year* cycle — a table of official land values that lagged the market badly. Because the framework prices were so far below market, they gave under-declarers cover: a contract written at the (low) framework price looked defensible.

**Land Law 2024** (effective August 2024) attacks this directly by **scrapping the rigid five-year framework and requiring provinces to publish annual land-price tables that track market values much more closely.** The logic: if the state's *own* reference price for a given street is current and realistic, then a contract declaring a wildly lower number stands out — it becomes far easier to flag, challenge, and tax on the true value. By raising the official benchmark toward reality, the law shrinks the gap that under-declaration hides in. (We have a whole sibling piece on the broader reform — see [Vietnam's Land Law 2024 and the coming property tax](/blog/trading/real-estate/vietnam-land-law-2024-and-the-coming-property-tax).) It's an elegant move: rather than chase millions of individual contracts, you fix the *reference price* and let the discrepancies surface themselves.

Now let's cross the Pacific to the system Vietnam might one day resemble.

## The US capital-gains model: basis, the gain, the rates

The United States taxes property sales on the opposite principle: it taxes the **gain**, the profit, and not the price. But that elegant principle is wrapped in so much relief — exclusions, deferrals, the step-up at death — that in practice a huge share of home-sale gains are never taxed at all. To understand it, we build up the calculation step by step, then meet each piece of relief in turn.

### Step one: find the cost basis

Recall the basis is "what it cost you." For Dana's home, bought at \$300,000:

- Start with the **purchase price**: \$300,000.
- Add **capital improvements** — lasting upgrades, not repairs. Dana spent \$20,000 on a new kitchen: basis rises to \$320,000.
- (You can also add certain **buying costs** like title fees; we'll fold a modest amount in for realism but keep the numbers round.)

Her **adjusted cost basis** is about \$320,000. Note what is *not* in the basis: the mortgage interest she paid over nine years, the property taxes, the cost of repainting a wall — all ordinary costs of owning, none of which raise the basis.

### Step two: compute the raw gain

$$\text{raw gain} = \text{sale price} - \text{adjusted basis} - \text{selling costs}$$

Dana sells for \$520,000. Her selling costs — a ~5–6% agent commission plus closing — come to about \$30,000. So:

$$\text{raw gain} = \$520{,}000 - \$320{,}000 - \$30{,}000 = \$170{,}000$$

That \$170,000 is her realized capital gain *before* any exclusion. In Vietnam, the tax base would have been the full \$520,000; in the US, it's this \$170,000 profit — already a far smaller number.

### Step three: apply the rate (long-term versus short-term)

The US rate depends critically on **how long you held the asset**:

- **Short-term capital gain** (held **one year or less**): taxed as ordinary income, at your regular income-tax bracket, which can run well above 30%. The system deliberately punishes quick flips.
- **Long-term capital gain** (held **more than one year**): taxed at preferential rates of **0%, 15%, or 20%**, depending on your total income. Most middle-income sellers land in the 15% bracket; very high earners hit 20% (and may owe an extra 3.8% net-investment-income tax on top).

This holding-period split is itself a behavioural lever: by taxing short-term gains at high ordinary rates and long-term gains at low preferential rates, the code nudges people to *hold for more than a year*. It's the first of several US features that reward holding and penalize selling quickly.

![A seven-step pipeline showing the US capital-gains calculation from sale price through subtracting basis, selling costs, and the exclusion to a taxable gain multiplied by the rate](/imgs/blogs/transfer-taxes-and-capital-gains-on-property-2.png)

The figure above walks the full calculation: sale price → subtract basis → subtract selling costs → raw gain → subtract the exclusion → taxable gain → apply the rate. Every subtraction shrinks the base; by the time you reach the rate, the number being taxed can be a small fraction of the price — or, thanks to the exclusion we're about to meet, zero. That's the structural opposite of Vietnam, where the tax base *is* the price and there are essentially no subtractions.

## The primary-residence exclusion: the biggest break most people never think about

Here is where the US system becomes genuinely generous to ordinary homeowners. The **primary-residence exclusion** lets you exclude — wipe out, never pay tax on — a large chunk of the gain on the sale of your *main home*:

- **\$250,000** of gain if you're single.
- **\$500,000** of gain if you're married filing jointly.

To qualify, you generally must have **owned and lived in the home as your primary residence for at least 2 of the last 5 years** before the sale (the "2-of-5" rule). It applies to your *home*, not to investment property or a vacation house you don't live in. And you can use it repeatedly — roughly once every two years — not just once in a lifetime.

The effect is enormous. Because most American homeowners' gains fall *under* \$250,000 (or \$500,000 for couples), the vast majority of home sales generate **no federal capital-gains tax at all**. The tax exists in principle, but the exclusion swallows it for ordinary households. This is, quietly, one of the largest tax subsidies in the US — and it's specifically a subsidy to *homeownership*, encouraging people to buy and stay in homes.

#### Worked example: Dana's \$300k→\$520k home sale (taxable gain of \$0)

Back to Dana and her husband, selling their main home of nine years. We computed a raw gain of \$170,000. Now apply the exclusion:

- **Raw gain:** \$170,000.
- **Married exclusion:** up to \$500,000 of gain excluded.
- **Taxable gain:** `\$170,000 − \$500,000 = negative → floored at \$0`.
- **Federal capital-gains tax owed:** `15% × \$0 = \$0`.

They keep the entire \$170,000 profit, tax-free, and don't even have to report the sale. Compare the *same* economic event in Vietnam: Minh, selling a flat at a similar gain, would pay 2% of the *whole price* — on a ₫7 billion sale, ₫140 million — regardless of his profit. The American homeowner with a six-figure gain pays nothing; the Vietnamese homeowner pays a real tax on a smaller (or even negative) profit.

*The US tax on a home sale looks fearsome on paper — up to 20% of your gain — but for most families the primary-residence exclusion quietly reduces it to zero, which is exactly why Americans treat their home as a tax-favored nest egg.*

But the exclusion has a hard edge that catches the wealthy and the long-tenured. If a single Californian bought a house decades ago for \$200,000 and sells today for \$1.4 million, the raw gain is \$1.2 million. The exclusion shelters \$250,000 of it; the **remaining \$950,000 is fully taxable** at 20% (plus the 3.8% surtax and state tax) — a federal bill north of \$190,000. The exclusion is a generous floor, not an unlimited shield, and on very large gains it barely dents the bill. (More on that misconception below.)

#### Worked example: a US gain that overflows the exclusion

Let's run that overflow case carefully, because it's where the US tax actually bites and where people get blindsided. Take a single owner — call him Ray — who bought a home for \$250,000, lived in it as his primary residence for fifteen years, put \$50,000 into improvements, and now sells for \$1,150,000 with \$70,000 of selling costs.

- **Adjusted basis:** `\$250,000 + \$50,000 = \$300,000`.
- **Raw gain:** `\$1,150,000 − \$300,000 − \$70,000 = \$780,000`.
- **Exclusion (single):** \$250,000.
- **Taxable gain:** `\$780,000 − \$250,000 = \$530,000`.
- **Federal tax at 20%:** `20% × \$530,000 = \$106,000`, plus a 3.8% net-investment-income surtax on much of it (~\$20,000) and state tax on top.

Ray walks away with a large profit, but the idea that "selling your home is tax-free" cost him a six-figure surprise, because his gain blew past the single-filer exclusion. Note how *every input mattered*: the \$50,000 of improvements and \$70,000 of selling costs shaved \$120,000 off the taxable gain (saving him ~\$24,000), and being married instead of single would have doubled his exclusion to \$500,000 and cut another \$50,000 off the bill.

*The US exclusion makes ordinary home sales tax-free, but on a large gain the part above the cap is taxed in full — so the people who benefited most from appreciation are exactly the ones who still owe, and the size of the bill turns on improvements, selling costs, and filing status they often overlook.*

## The 1031 exchange: defer the gain forever

The exclusion handles your *home*. For *investment* property — a rental, a commercial building, raw land held for profit — the US offers a different and even more powerful tool: the **1031 like-kind exchange**, named after Section 1031 of the tax code.

The idea: instead of *selling* an investment property (which realizes the gain and triggers the tax), you **exchange** it for another investment property of "like kind." When you do, the tax law treats it as if you never cashed out — your old **cost basis carries over** to the new property, and the gain is **deferred**, not taxed, until some later sale that *isn't* a 1031 exchange. Because real estate is broadly "like kind" to other real estate, you can roll an apartment into an office building into a warehouse into a strip mall, deferring the gain at every step.

There are strict rules that make it a genuine exchange rather than a disguised sale:

- **The 45-day identification rule:** within 45 days of selling the old property, you must *identify in writing* the replacement property (or properties) you intend to buy.
- **The 180-day closing rule:** you must *close* on the replacement within 180 days of the sale.
- **A qualified intermediary** must hold the sale proceeds in between — you're never allowed to touch the cash, or it counts as a sale.
- The replacement must be of equal or greater value (any cash you pull out, called "boot," is taxable).

The strategy investors describe as **"swap till you drop"** chains these exchanges across a lifetime: every time you'd otherwise sell and pay tax, you 1031 into a bigger or better property instead, carrying the deferred gain forward, never paying. The deferred tax compounds in your favour — money you'd have paid the government instead keeps working in the next property. And then comes the part that turns deferral into outright elimination.

![A timeline of the 1031 exchange chain: sell property A on day zero, identify a replacement within 45 days, close within 180 days with the gain deferred, repeat the swap over years, then a step-up at death resets the basis and erases the gain](/imgs/blogs/transfer-taxes-and-capital-gains-on-property-3.png)

The timeline above shows the full chain: sell A (gain locked but untaxed) → identify B within 45 days → close on B within 180 days (basis carries over, gain deferred) → repeat the swap for years → and finally, at death, a **step-up in basis** resets everything and wipes the deferred gain away. That last step is the magic, and it deserves its own section.

#### Worked example: a 1031 deferral on Dana's rental

Dana also owns a rental condo, bought for \$200,000, now worth \$400,000 — an unrealized gain of \$200,000. If she simply *sells* it, she realizes the gain and owes long-term capital-gains tax: `15% × \$200,000 = \$30,000` (more if she's in the 20% bracket, plus depreciation recapture, which we'll set aside for clarity).

Instead, she does a 1031 exchange. She sells the condo for \$400,000, a qualified intermediary holds the cash, within 45 days she identifies a \$420,000 duplex, and within 180 days she closes on it. Result:

- **Tax owed today:** \$0. The \$200,000 gain is *deferred*, not forgiven.
- **Her basis in the new duplex:** her old \$200,000 basis carries over (adjusted for the extra \$20,000 she put in), not the \$420,000 she "paid." So the deferred gain rides along, embedded in a low basis.
- **The \$30,000 she didn't pay** stays invested in the larger property, compounding.

She can repeat this indefinitely. *The 1031 exchange doesn't erase the gain — it lets the tax-deferred gain ride forward into ever-larger properties, so the investor compounds on money they'd otherwise have handed to the government, sometimes for the rest of their life.*

There's one wrinkle on investment property worth naming so the picture is honest: **depreciation recapture.** While Dana owned her rental, the US tax code let her *depreciate* the building — deduct a slice of its cost each year against her rental income, as if the structure were wearing out. Those deductions lowered her taxable rental income year after year (a real benefit). But they also quietly *lowered her basis* — depreciation reduces the basis the same way it shielded income. When she eventually sells *without* a 1031 exchange, the portion of her gain that corresponds to depreciation she took is "recaptured" and taxed at a special rate (up to 25%), separate from the regular capital-gains rate. The 1031 exchange defers this recapture too, rolling it forward with the rest of the gain. The lesson: a rental's tax story is a *two-part* deferral — the appreciation gain and the depreciation recapture — and both ride forward in a 1031 and both can be erased by the step-up at death. It's part of why investment real estate is so tax-favoured in the US, and part of why critics call the whole structure a standing subsidy to property capital.

## Step-up in basis at death: the gain that disappears

Now the feature that makes "swap till you drop" literal. When an owner **dies**, the cost basis of their property is **reset — "stepped up" — to its fair-market value on the date of death** in the hands of their heirs. The heirs inherit the property as if they bought it that day at full current value. All the unrealized (and deferred) gain that built up during the owner's life simply *vanishes* for income-tax purposes.

**Step-up in basis** is the most consequential — and most criticized — feature of the whole system. Walk through what it does to our chain:

- Dana bought rentals for a total basis of, say, \$200,000 and, through decades of 1031 exchanges, now holds property worth \$2,000,000. She's carried a \$1.8 million deferred gain the whole way, never paying tax on it.
- If she sold the day before she died, she'd owe capital-gains tax on the \$1.8 million gain — perhaps \$360,000+ federal.
- Instead, she dies holding it. Her heirs' basis is **stepped up to \$2,000,000** — the market value at death. The \$1.8 million of lifetime gain is *erased* for capital-gains purposes.
- If the heirs sell the next week for \$2,000,000, their gain is `\$2,000,000 − \$2,000,000 = \$0`. No income tax on a lifetime of appreciation.

This is how the genuinely wealthy can hold appreciating real estate for generations and pay *zero* capital-gains tax on it: defer with 1031 exchanges while alive, then let the step-up erase the deferred gain at death. (A separate *estate* tax can apply to very large estates above a high threshold, but that's a different tax from the income tax on the gain, and most estates fall under the exemption.) The income-tax gain — the thing this whole article is about — is gone.

#### Worked example: the lock-in, an owner who refuses to sell

Step-up creates a powerful reason *not to sell*, which brings us to the central behavioural effect of gain-based taxes. Picture an elderly American widow who bought her home in 1985 for \$120,000. It's now worth \$920,000 — an \$800,000 unrealized gain. She's single, so her exclusion is only \$250,000.

- **If she sells now:** taxable gain = `\$800,000 − \$250,000 = \$550,000`; tax at 20% (plus surtax and state) easily exceeds **\$130,000**.
- **If she holds until death:** her heirs' basis steps up to \$920,000; the entire \$800,000 gain is erased; tax on it is **\$0**.

The tax code is, in effect, paying her \$130,000+ to *not sell* — to stay in a house that may be far too big for her, and to keep it off the market. She'd be irrational to sell. *The combination of a taxable gain on sale and a tax-free step-up at death gives long-tenured owners a six-figure reason to never move, which is the lock-in effect in one household.*

Multiply that household by millions and you get a market-wide phenomenon.

## The lock-in effect: taxes that freeze sellers in place

The **lock-in effect** is the tendency of a capital-gains tax to *discourage owners from selling*, because selling triggers a tax that holding defers. The bigger the embedded gain, the bigger the tax bill on sale, and the stronger the incentive to just... stay put. It's the single most important *market-level* consequence of taxing gains rather than prices, and it works through a clean causal chain.

![A causal graph showing a high gains tax leading owners to not sell their home and not trade up, which reduces inventory and tightens the market](/imgs/blogs/transfer-taxes-and-capital-gains-on-property-5.png)

The figure above traces the mechanism. A high gains-on-sale tax (1) deters owners with large unrealized gains from selling their home, and (2) deters them from "trading up" — because moving to a bigger house means selling the current one first and paying the tax now. Both behaviours mean homes that *would* have come to market stay off it. (3) Inventory shrinks; (4) with fewer listings, the market tightens — turnover slows, and prices on the homes that *do* sell get propped up by scarcity. The tax meant to raise revenue ends up *restricting supply*.

This isn't theoretical. In the US, the lock-in effect is amplified by a second mechanism — the **"mortgage rate lock-in"** — where owners who locked in 3% mortgages in 2020–21 won't sell into a 6.5–7% rate environment because they'd lose their cheap loan. Stack the capital-gains lock-in on top of the rate lock-in and you get the frozen US resale market of 2023–2025: existing-home sales fell to multi-decade lows, not because nobody wanted to move, but because the *cost* of moving — in tax and in lost cheap debt — was too high. The supply of existing homes dried up, which (perversely) kept prices high even as affordability collapsed. (For the broader machinery of how supply scarcity drives prices, see [supply elasticity](/blog/trading/real-estate/supply-elasticity-why-some-cities-boom-and-bust).)

Vietnam's system produces a *different* freeze. Because the 2% tax attaches to the *transaction* and not the gain, it doesn't lock in long-term holders the way the US tax does — there's no step-up to wait for. Instead, Vietnam's tax (combined with high transaction costs generally) discourages *frequent trading* and *flipping*: each round-trip into and out of a property costs ~2.5% on the way out plus the registration fee on the way in, which eats into a flipper's margin. So where the US tax freezes long-tenured owners in place, Vietnam's tax discourages rapid turnover — and, as we saw, pushes the dishonest part of the market into under-declaration rather than into holding. Different design, different distortion. (For how taxes more broadly shape speculation and hoarding behaviour, see [how tax shapes behaviour](/blog/trading/real-estate/how-tax-shapes-behavior-speculation-and-hoarding).)

![A bar chart comparing the tax owed on the same home sale under Vietnam's 2% price-based tax versus the US capital-gains tax with and without the primary-residence exclusion](/imgs/blogs/transfer-taxes-and-capital-gains-on-property-4.png)

The bar chart above puts numbers on the comparison for one illustrative sale: a \$280,000 home with an \$80,000 gain. Vietnam's 2% PIT takes \$5,600 — off the *price*, owed even if the gain were zero. The US gain tax *without* the exclusion would take \$12,000 (15% of the \$80,000 gain) — *more*, because this owner had a healthy profit. But *with* the primary-residence exclusion, the \$80,000 gain is fully sheltered and the US tax drops to **\$0**. That spread — \$12,000 down to \$0 — is the exclusion doing its work, and it's why "how much does the state take when you sell?" has no single answer: it depends entirely on the *design*, the profit, and whether relief applies.

## Common misconceptions

Property tax-on-sale is a minefield of confident wrong beliefs. Here are the ones that cost people the most.

### "You only pay tax if you made a profit" — false in Vietnam

This is true in a gain-based system like the US and **flatly false in Vietnam**. Vietnam's 2% PIT is computed on the *transfer price*, not the gain, so you owe it even when you sell at a loss. The seller who bought at the 2022 peak and is dumping a flat below cost in 2024 still pays 2% of whatever they sell for. Many sellers are genuinely shocked by this — they assume "no profit, no tax," which is the gain-based intuition, and it simply doesn't apply. If you sell property in Vietnam, budget the 2% regardless of how the trade went.

### "The primary-residence exclusion covers everything"

It covers a lot — \$250k single / \$500k married — but it is a *cap*, not a blanket. Gains *above* the exclusion are fully taxable. A long-tenured owner in an expensive coastal US city can easily have a \$1 million+ gain, of which only \$250k–\$500k is sheltered; the rest is taxed at 15–20% plus surtaxes. And the exclusion only applies to your *primary residence* (the 2-of-5-years rule) — not to a rental, a flip, or a second home. People who assume their home sale is automatically tax-free can get a nasty surprise on a very large gain or a property that doesn't meet the residence test.

### "The 1031 exchange is a loophole that erases the tax"

The 1031 exchange **defers** the gain; it doesn't, by itself, erase it. If you ever sell *without* doing another exchange, all the deferred gain comes due at once. What *erases* the gain is the separate **step-up in basis at death** — and it's the *combination* (defer with 1031 while alive, erase with step-up at death) that lets gains escape income tax entirely. Calling 1031 alone a "tax eraser" misunderstands the mechanism: it's a deferral tool, powerful precisely because deferral compounds and because death may eventually convert deferral into elimination.

### "Under-declaring the price is harmless because everyone does it"

It's common, but it is **tax evasion**, and the risk falls heaviest on the *buyer*, who ends up with legal proof of paying far less than they actually did. If the deal sours, the buyer can typically only recover the declared amount. Add back-taxes, penalties, and (with Land Law 2024's realistic price tables) a rising chance of detection, and the ₫75 million "saved" can turn into a far larger loss. "Everyone does it" is not a legal defense, and the exposure is asymmetric and real.

### "Transfer tax and capital-gains tax are basically the same thing"

They are structurally opposite. A transfer tax is a percentage of the **price** and is owed regardless of profit; a capital-gains tax is a percentage of the **profit** and is zero if you didn't profit. They reward and punish completely different behaviours — one discourages *transacting*, the other discourages *realizing/selling* — and conflating them leads to wrong predictions about how each market behaves. Keep the base straight (price vs. gain) and everything else follows.

### "Selling costs and improvements don't matter for tax"

In a gain-based system they matter a great deal: every dollar of legitimate capital improvement *raises your basis*, and every dollar of selling cost *lowers your gain* — both shrink the taxable amount. Dana's \$20,000 kitchen and \$30,000 in selling costs reduced her gain by \$50,000 and thus her *potential* tax by \$7,500 at 15%. Keeping receipts for improvements is real tax planning. In Vietnam's price-based system, by contrast, none of this matters — improvements and selling costs don't reduce the 2%-of-price tax at all, which is one more way the systems diverge.

## How it shows up in real markets

The abstractions above leave fingerprints all over real markets. Here are concrete, named instances.

### Vietnam: a market that runs on two prices

Walk into any Vietnamese real-estate transaction and you'll often find *two* prices in play: the real one the parties agreed, and the lower one written for tax. Under-declaration has been so widespread that it corrupts the country's housing-price data — making it genuinely hard for policymakers, banks, and researchers to know what homes actually trade for. The Ministry of Finance and tax authorities have for years tried to crack down: tax offices in HCMC and Hanoi periodically reject suspiciously low declarations and demand re-filing at market-consistent prices, and there have been high-profile cases of contracts bounced back for under-declaration. The structural fix — raising the state's *own* reference prices so lowball contracts stick out — is exactly what **Land Law 2024** does by replacing the stale five-year price framework with annual, market-tracking land-price tables. Whether it meaningfully shrinks the under-declaration gap is one of the most-watched questions in Vietnamese property policy as of 2026.

### The US "great lock-in" of 2023–2025

After the Federal Reserve drove mortgage rates from a record-low **2.65% (January 2021)** to a peak around **7.79% (October 2023)**, the US resale market froze. Roughly two-thirds of US mortgage holders had rates below 4%; selling meant giving up that cheap loan *and*, for long-tenured owners with big gains, triggering a capital-gains bill once their gain exceeded the exclusion. The two lock-in forces compounded: existing-home sales fell to their lowest levels in nearly three decades, inventory stayed scarce, and — counter to what you'd expect when affordability collapses — prices barely fell, because so few homes came to market. The capital-gains lock-in (taxes that punish selling, step-up that rewards holding) was a real part of why supply stayed frozen even as buyers were priced out. It's the lock-in figure from this article, playing out across a whole country.

### The 1031 exchange and the American real-estate fortune

A striking share of multi-generational US real-estate wealth is built on the defer-then-step-up chain. Investors acquire a small rental, ride its appreciation, 1031 into a larger property, and repeat — sometimes a half-dozen times over decades — never paying capital-gains tax along the way. When they die, the step-up resets the basis and erases the lifetime gain for their heirs. Critics across the political spectrum have repeatedly proposed limiting 1031 exchanges or curbing the step-up (the step-up's "loss" to the Treasury is estimated in the tens of billions of dollars a year), precisely because the combination lets large gains escape income tax entirely. That the rules have survived decades of proposed reform is itself evidence of how entrenched — and how valued by property owners — the deferral machinery is.

### California's Proposition 13 and lock-in by another route

A cousin of the capital-gains lock-in shows up in *annual* property taxes too. California's **Proposition 13** caps how fast a home's assessed value (and thus its annual property tax) can rise *as long as you keep owning it* — but reassesses to full market value when it sells. The result: long-tenured owners pay tiny property taxes on homes worth millions, and selling would reset that to a far higher bill. Like the capital-gains lock-in, it pays people to stay put, freezing inventory and contributing to California's chronic housing shortage. Different tax, same lesson: a tax that resets *on sale* and is frozen *while you hold* manufactures lock-in. (For how the annual hold-side taxes fit into the full picture, see the [full map of how real estate is taxed](/blog/trading/real-estate/how-real-estate-is-taxed-the-full-map).)

### Singapore's Seller's Stamp Duty: a transfer tax aimed at speculation

Singapore offers a clean example of a *price-based* tax deliberately tuned to shape behaviour. Its **Seller's Stamp Duty** charges a steep tax on residential property *sold within a few years of purchase* — high if you flip within the first year, declining to zero if you hold past the threshold. It's a transfer tax (a percentage of the sale price) explicitly designed to *punish quick flipping* and cool speculative churn, much as Vietnam's flat-but-time-blind 2% raises the cost of every round-trip. The contrast with the US is instructive: the US uses a *holding-period* split (short-term vs. long-term gains) to discourage flips within the gain-based system, while Singapore bolts a flip-penalty onto the price-based one. Same goal — slow speculation — via opposite tax bases.

![A matrix comparing Vietnam and the United States on the base, rate, main relief, and the behaviour each sale tax encourages](/imgs/blogs/transfer-taxes-and-capital-gains-on-property-7.png)

The matrix above is the whole article on one screen. Vietnam: base is the *price*, rate is a flat 2% (plus 0.5%), relief is almost none, and the behaviour it encourages is *less trading and more under-declaration*. The US: base is the *gain*, rate is 0/15/20%, relief is generous (the \$250k/\$500k exclusion plus the 1031 deferral and step-up), and the behaviour it encourages is *less selling* — the lock-in. Read the rows and you can predict each market's pathology from its tax design alone.

## When this matters to you / further reading

This stops being abstract the moment you're about to sell — or buy from someone who's about to sell.

If you're **selling property in Vietnam**, the practical upshot is simple and slightly grim: budget the **2% PIT plus 0.5% registration fee** (about ₫175 million on a ₫7 billion sale), and budget it *whether or not you made money*, because the tax is on the price, not the profit. Treat the agent's suggestion to under-declare as what it is — illegal, and a risk that falls hardest on the buyer's legal protection — and weigh it against rising detection under Land Law 2024's realistic price tables. Keep your own honest record of what you paid and received; if Vietnam ever shifts toward a true gain-based tax (or the floated annual property tax arrives), that record becomes valuable.

If you're **selling a home in the US**, the upshot is more hopeful: compute your *gain*, not your price, subtract your improvements and selling costs (keep those receipts), and check the **2-of-5-years** residence test — for most families the **\$250k/\$500k exclusion** drops the federal tax to zero. The trap is the *large* gain: if your profit exceeds the exclusion (common in long-held or high-cost-city homes), the excess is taxed at 15–20% plus surtaxes, and that's where planning — timing, the residence test, or for investment property a 1031 exchange — actually moves the number.

And if you're trying to *understand a market* rather than transact in one, the deepest lesson is the one the matrix captured: **the base of the tax — price versus gain — predicts the market's behaviour.** Tax the price and you discourage trading and invite under-declaration; tax the gain and you discourage selling and manufacture lock-in. The state's cut is never just revenue; it's a set of incentives quietly bent into the market's structure.

To go deeper into the surrounding machinery:

- The full picture of *every* property tax — at purchase, while holding, on rental income, and at sale — is in [how real estate is taxed: the full map](/blog/trading/real-estate/how-real-estate-is-taxed-the-full-map).
- How taxes more broadly steer speculation, flipping, and hoarding is in [how tax shapes behaviour](/blog/trading/real-estate/how-tax-shapes-behavior-speculation-and-hoarding).
- The Vietnamese reform driving the attack on under-declaration — and the property tax that may be coming — is in [Vietnam's Land Law 2024 and the coming property tax](/blog/trading/real-estate/vietnam-land-law-2024-and-the-coming-property-tax).
- And because transaction taxes are a big part of why owning is more expensive than it looks, see how they fold into the buy-versus-rent decision in [rent vs buy: the real math](/blog/trading/real-estate/rent-vs-buy-the-real-math).

The next time you sell something — a home, a plot, a building — and the state takes its slice, you'll know exactly *what* it's taking the slice of, *why* the rule is built that way, and how that one design choice ripples out into the whole market around you.
