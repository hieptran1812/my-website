---
title: "Fiscal Policy for Traders: Spending, Deficits, and Aggregate Demand"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "A beginner-friendly deep dive into fiscal policy — government spending, taxes, the deficit, the multiplier, and crowding out — and why, when monetary policy is stuck at the zero bound, the Treasury becomes the dominant macro force that drives demand, inflation, and the supply of bonds."
tags: ["macro", "fiscal-policy", "government-spending", "deficits", "aggregate-demand", "fiscal-multiplier", "crowding-out", "bond-supply", "inflation", "trading"]
category: "trading"
subcategory: "Macro Trading"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — Fiscal policy — government spending and taxation — is the other half of macro, and traders ignore it at their peril. It drives aggregate demand directly (not indirectly through the price of money the way monetary policy does), it sets the deficit, and the deficit governs the supply of bonds and the path of growth and inflation. When monetary policy is pinned at the zero bound, fiscal policy becomes the dominant macro force — and 2020-21 proved it.
>
> - **Fiscal = spending minus taxes.** When the government spends more than it taxes, it runs a **deficit** and must borrow the gap by selling bonds. When it taxes more than it spends, it runs a **surplus**. The deficit is the single number that ties spending, demand, and bond supply together.
> - **The fiscal multiplier** is why \$1 of spending can lift GDP by more than \$1: the dollar becomes someone's income, a fraction gets re-spent, and the chain sums to roughly \$1.50 in a slack economy. That is how a spending program moves *aggregate demand*.
> - **Crowding out is conditional, not automatic.** A deficit competes for savings and lifts yields only when the economy is at full employment. In a slump with idle capacity — like 2020 — the same deficit can run with the 10-year Treasury still near 1%.
> - The one number to remember: the **2020 federal deficit hit 15.0% of GDP — \$3.13 trillion** — the largest US peacetime deficit ever, and it dwarfed anything monetary policy could do once the policy rate was already at zero.

In the spring of 2020 the world's central banks did everything they knew how to do, and it was not enough. The Federal Reserve cut its policy rate to essentially zero on March 15, restarted bond-buying, and opened a dozen emergency facilities. That is the entire conventional and unconventional monetary toolkit, fired at once. And yet the thing that actually put money into tens of millions of household checking accounts that spring was not the Fed at all. It was the United States Congress, writing checks.

The CARES Act, signed March 27, 2020, was worth about \$2.2 trillion. It sent direct payments to households, topped up unemployment benefits by \$600 a week, and stood up the Paycheck Protection Program for small businesses. A second relief bill in December added about \$0.9 trillion. The American Rescue Plan in March 2021 added another \$1.9 trillion. Add it up and you get roughly \$5 trillion of *fiscal* support in twelve months. The federal deficit for fiscal 2020 came in at **\$3.13 trillion — 15.0% of GDP** — the largest peacetime deficit in American history. For comparison, the entire monetary response, however dramatic it looked on a chart, was a rate cut that ran out of room at zero and a balance sheet expansion that, by design, does not put a single dollar into anyone's pocket. It swaps one safe asset for another.

This is the lesson most newcomers to macro learn too late: **fiscal policy is the other half of macro, and at certain moments it is the dominant half.** Monetary policy gets all the attention — every word the Fed chair says is parsed to death — but monetary policy works *indirectly*, by changing the price of money and hoping people borrow and spend. Fiscal policy works *directly*: the government spends, and that spending is demand, immediately, no borrowing decision required from anyone else. When the indirect lever is jammed against the zero bound, the direct lever is the only one with room left. If you cannot read fiscal policy — the spending, the deficit, the bond supply it creates, the demand it injects — you are reading macro with one eye closed. By the end of this post you will be able to read it clearly, and we build every concept from zero.

![Two columns contrasting the monetary lever and the fiscal lever on the economy](/imgs/blogs/fiscal-policy-for-traders-spending-deficits-demand-1.png)

The figure above frames the whole post. On the left is the **monetary lever**, worked by the central bank: it changes the *price* of money (the policy rate) and the *quantity* of money in the banking system (QE and QT), and it works only if households and firms respond by borrowing and spending more. On the right is the **fiscal lever**, worked by the Treasury and Congress: it changes demand *directly* by spending money and by taxing it away. The deep difference — the one that makes 2020 make sense — is in the bottom rows. Monetary policy at zero is *out of room*; fiscal policy still has plenty. And fiscal policy leaves something behind that monetary policy does not: a **deficit**, which means a pile of new bonds the Treasury has to sell. Hold that contrast in your head; everything below fills it in.

## Foundations: spending, taxes, the deficit, and the multiplier

Before any of the trading insight lands, you need a handful of plain ideas, defined from scratch. None of them require finance background. If you have ever run a household budget, you already have the intuition; fiscal policy is the same idea at the scale of a government.

### What "fiscal policy" actually means

**Fiscal policy** is the government's use of its *spending* and its *taxation* to influence the economy. That is the entire definition. Two knobs: how much the government spends, and how much it collects in taxes. Everything else — deficits, debt, multipliers, crowding out — is a consequence of turning those two knobs.

Contrast this with **monetary policy**, which is the *central bank's* management of the price and quantity of money — interest rates, the size of its balance sheet. The two are run by different institutions for a reason. In the US, monetary policy is run by the Federal Reserve, which is deliberately insulated from elected politicians so it can make unpopular decisions (like raising rates into a slowdown). Fiscal policy is run by Congress and the President — the elected, political branches — because deciding *what* to spend money on and *whom* to tax is an inherently political choice. A road, a missile, a stimulus check, a tax cut: those are decisions a democracy makes through its legislature, not decisions a technocratic committee makes behind closed doors.

A useful everyday analogy: think of the economy as a household and the government as one very large member of it. Monetary policy is like the *interest rate on the household's credit card* changing — it makes borrowing cheaper or dearer, which *nudges* everyone's spending decisions but does not directly buy anything. Fiscal policy is the large family member *actually going to the store and spending*, or *handing cash to the others*, or *taking some of their paychecks*. One changes the incentives around money; the other moves money itself.

### Government spending versus taxation

The two knobs work in opposite directions on demand.

**Government spending** (often written `G` in textbooks) is money the government pays out: salaries for soldiers and teachers, contracts to build roads and bridges, purchases of equipment, and — a huge category — **transfer payments** like Social Security, Medicare, unemployment benefits, and stimulus checks. Transfer payments are special: the government is not buying anything, it is simply moving money to households, who then spend it. Either way, government spending *adds* to demand in the economy.

**Taxation** (`T`) is money the government takes in: income taxes, payroll taxes, corporate taxes, tariffs. Taxes *remove* money from households and firms, which *subtracts* from demand — every dollar paid in tax is a dollar not spent at a store.

So the net effect of fiscal policy on demand is, roughly, **spending minus taxes**. When the government spends more than it taxes, it is net *adding* demand. When it taxes more than it spends, it is net *removing* demand. This simple subtraction is the engine of the whole subject, and it has a name.

### The budget balance: deficit and surplus

The **budget balance** is just taxes minus spending in a given year.

- If taxes exceed spending, the government runs a **surplus** — it took in more than it paid out. (The last US federal surplus was in fiscal 2001.)
- If spending exceeds taxes, the government runs a **deficit** — it paid out more than it took in. This is the normal state of affairs for most governments most of the time.

When the government runs a deficit, it has to fund the gap somehow. A household that spends more than it earns puts the difference on a credit card or takes a loan. A government does the same thing, but its "credit card" is the **bond market**: it sells Treasury securities — bills, notes, and bonds — to investors, who lend it the money in exchange for a promise of repayment with interest. So a deficit is not a paper concept. **A \$3.13 trillion deficit is \$3.13 trillion of new bonds the Treasury must sell into the market that year.** This is the bridge from fiscal policy to your screen: the deficit is the faucet for bond supply, and bond supply is one of the things that moves yields. We will return to this hard.

The accumulation of all past deficits, minus all past surpluses, is the **national debt** — the total amount the government owes. The 2020 deficit added to a debt that has since climbed past \$37 trillion (about 123% of GDP including debt held within government accounts). A deficit is a *flow* (per year); the debt is the *stock* (the running total). Keep the two straight: people constantly confuse the annual deficit with the total debt, and they are off by a factor of more than ten.

### The deficit in context: how big is big?

Raw dollar figures are hard to feel, so economists almost always express the deficit as a **percentage of GDP** — the deficit divided by the total size of the economy that year. This normalizes for inflation and for the economy growing over time: a \$1 trillion deficit means something very different in a \$10 trillion economy than in a \$25 trillion one.

![Bar chart of the US federal deficit as a percent of GDP from 2019 to 2025](/imgs/blogs/fiscal-policy-for-traders-spending-deficits-demand-2.png)

The chart above is the single most important fiscal picture in this post. It shows the US federal deficit as a percent of GDP from 2019 through 2025. Read the bars left to right. In 2019, a fairly normal year, the deficit was **4.6% of GDP** — already on the high side historically, but unremarkable. Then 2020 happens: the deficit explodes to **15.0% of GDP**, a level the United States had not seen since World War II. 2021 stays enormous at 12.3% as the American Rescue Plan lands. Then the deficit settles into a new, structurally higher range — 5.4%, 6.2%, 6.3%, 5.9% — through 2025.

There is a green dashed line at roughly 3%. That is a rule of thumb economists cite for the deficit level that roughly *stabilizes* the debt-to-GDP ratio when the economy is growing at its normal pace: if the deficit is around 3% and nominal GDP grows around 4-5%, the debt grows slower than the economy and the debt ratio holds steady or falls. Notice that the US has been running *well above* that line every single year shown, even in the "calm" post-COVID years. That structurally large deficit is itself a live macro story — it means persistent, heavy bond issuance, which is a standing tailwind for yields and a reason term premium has crept back into the long end. We come back to that in the playbook.

For now, absorb the headline: **2020's 15.0%-of-GDP deficit was a once-in-a-generation fiscal event**, and it is the case study that runs through the rest of this post.

### The same picture in dollars — and why dollars are what the bond market sees

The percent-of-GDP view is the right way to *compare* deficits across years and across countries. But the bond market does not buy "percent of GDP" — it buys a specific number of dollars of bonds. So it is worth seeing the same deficit in raw dollars, because that dollar figure is, to a first approximation, the amount of *new bonds the Treasury has to sell that year*.

![Bar chart of the US federal deficit in dollars from 2019 to 2025](/imgs/blogs/fiscal-policy-for-traders-spending-deficits-demand-4.png)

The chart is the dollar twin of the percent chart. In 2019 the deficit was about \$0.98 trillion. In 2020 it leapt to **\$3.13 trillion** — the red bar — then \$2.78 trillion in 2021, before settling into a \$1.4-to-\$1.8 trillion range through 2025. Now read it as a *financing* problem rather than an *economics* problem. To run a \$3.13 trillion deficit, the Treasury had to find \$3.13 trillion of lenders in a single fiscal year, on top of *rolling over* the trillions of existing debt that matured and needed refinancing. The gross amount of Treasury securities auctioned in 2020 ran into the tens of trillions once you count the rollovers. That is the supply the bond market had to swallow — and it swallowed it at sub-1% yields, which, as we will see, is only possible in a deeply slack economy with the central bank as a backstop buyer.

This is the single most useful mental move in fiscal trading: **translate every deficit number into a bond-supply number.** A widening deficit is a rising supply of bonds; a narrowing deficit is a falling supply. When you read a CBO projection that the deficit will average \$2 trillion a year for the next decade, do not file it under "politics" — file it under "\$2 trillion a year of net new Treasury duration the market must absorb, every year, indefinitely." That standing supply is a structural force on the long end of the curve, and it is *entirely* a fiscal phenomenon.

## Expansionary versus contractionary fiscal policy

With the knobs defined, the two *directions* of fiscal policy follow immediately.

**Expansionary fiscal policy** means turning the knobs to *add* demand: spending more, taxing less, or both. The deficit widens. Governments do this when the economy is weak — in a recession or a slowdown — to prop up demand and employment. The 2020-21 response was expansionary fiscal policy at maximum volume: a torrent of spending and direct payments, deliberately blowing the deficit out to 15% of GDP, precisely *because* the economy was collapsing and needed demand injected fast.

**Contractionary fiscal policy** means turning the knobs to *remove* demand: spending less, taxing more, or both. The deficit narrows (or a surplus grows). Governments do this — or are forced to do this — when an economy is overheating, when inflation is too high, or when the debt load has become unsustainable. **Austerity** is the political term for aggressive contractionary fiscal policy: cutting public spending and raising taxes to shrink the deficit, usually painfully. Europe's response to its 2010-2012 debt crisis was austerity, and the lesson many economists drew was that contractionary fiscal policy *in a weak economy* deepened the weakness — a cautionary tale we will see again under the multiplier.

Here is the trader's framing of this distinction. Expansionary fiscal is a demand *tailwind*: it supports growth, supports corporate revenues, tends to be reflationary (pushing inflation up), and — crucially — *increases* bond supply because the deficit widens. Contractionary fiscal is a demand *headwind* — often called **fiscal drag** — it cools growth, can be disinflationary, and *reduces* bond supply because the deficit narrows. So when you read that a government is "tightening fiscal policy" or that a stimulus is "rolling off," you should immediately translate: less demand support, potential drag on growth, and less new bond supply ahead.

One subtlety that trips people up: it is not the *level* of the deficit that adds or subtracts demand at the margin, but the *change* in it. A deficit that holds steady at 6% of GDP is not adding *new* demand year over year — it is the *widening* from, say, 4% to 6% that injects demand, and the *narrowing* from 6% back to 4% that drags. That change — the year-over-year shift in the deficit — has a name traders use constantly: the **fiscal impulse**. A positive fiscal impulse (a widening deficit) is expansionary at the margin; a negative impulse (a narrowing deficit) is contractionary at the margin. We will make this precise in the playbook, because the impulse, not the level, is what you actually trade.

## Automatic stabilizers versus discretionary policy

Not all fiscal expansion is a *decision*. A huge amount of it happens on autopilot, and distinguishing the two is essential to reading the deficit correctly.

**Automatic stabilizers** are parts of the budget that swing toward expansion in a downturn and toward contraction in a boom *without anyone passing a law*. They are baked into the structure of taxes and spending. Two big examples:

- **Taxes fall automatically in a recession.** Income and payroll tax revenue is a roughly fixed percentage of wages and profits. When the economy shrinks, wages and profits shrink, so tax revenue falls — automatically — even with no change in tax *rates*. Lower taxes means the government is removing less demand, which cushions the fall.
- **Spending rises automatically in a recession.** Unemployment benefits, food assistance, and other "safety net" programs pay out more when more people are out of work — automatically, because more people qualify. More spending means the government is adding more demand, again cushioning the fall.

So even if Congress did *absolutely nothing* in a recession, the deficit would widen on its own: revenue drops while safety-net spending rises. This is by design — it is a built-in shock absorber that makes downturns less severe without requiring slow, contentious legislation. In a boom it runs in reverse: revenue surges and safety-net spending falls, automatically *narrowing* the deficit and cooling the economy.

**Discretionary fiscal policy** is the part that *does* require a decision: Congress voting to pass a stimulus bill, cut tax rates, fund an infrastructure program, or start a war. The CARES Act was discretionary — a deliberate, legislated \$2.2 trillion injection on top of whatever the automatic stabilizers were already doing.

Why does this distinction matter to a trader? Two reasons. First, **discretionary policy is forecastable** in a way that pure automatic stabilizers are not — bills are debated for weeks in public, so you can position ahead of a stimulus package or a spending cliff. The market often prices the fiscal impulse as the legislative odds shift. Second, when you see the deficit widen, you should ask *which kind* of widening it is. A deficit widening because of automatic stabilizers in a recession is a *symptom* of a weak economy (and a built-in cushion). A deficit widening because of a discretionary stimulus is a deliberate *injection*. They have different implications: the first tends to coincide with falling yields (weak economy, the flight to safety), the second can push yields up (more supply, more demand, more inflation risk). The COVID deficit was both at once — automatic stabilizers blew out *and* a historic discretionary package landed on top — which is exactly why it reached an unprecedented 15% of GDP.

## The fiscal multiplier and aggregate demand

Now the central mechanism — the one that explains *why* government spending moves the whole economy, and by how much. To get there we first need the phrase that fiscal policy ultimately acts on.

### Aggregate demand

**Aggregate demand** is the total spending in an economy over a period — all the demand for goods and services added together. It has four classic components: **C**onsumption (household spending), **I**nvestment (business spending on plant, equipment, housing), **G**overnment spending, and net e**X**ports (exports minus imports). Economists write it as `AD = C + I + G + NX`.

Two things to notice. First, `G` — government spending — is *literally one of the four components*. When the government spends, aggregate demand rises by that amount mechanically, before any further effects. Fiscal policy does not have to persuade anyone to do anything; it *is* a direct slug of demand. That is the directness we keep emphasizing. Second, government policy can also move the *other* components: a tax cut raises households' after-tax income, which lifts `C`; a stable, supportive fiscal environment can encourage businesses to invest, lifting `I`. So fiscal policy reaches aggregate demand both directly (through `G`) and indirectly (through `C` and `I`).

When you read that a recession is a "shortfall of aggregate demand," this is the framework: households and businesses have pulled back (`C` and `I` collapsed), and the question is whether `G` can be cranked up enough to fill the hole. In 2020, `C` and `I` cratered as the economy locked down, and the entire fiscal response was an attempt to plug that crater with `G` and with transfers that would revive `C`.

### The multiplier: why \$1 can become more than \$1

Here is the deep idea. When the government spends a dollar, that dollar does not just vanish into the economy and stop. It becomes *someone's income*, and that someone spends part of it, which becomes the *next* person's income, who spends part of *that*, and so on, in a chain that ripples outward. The total effect on GDP is the sum of the whole chain — which can be *larger* than the original dollar. That ratio — total change in GDP divided by the initial change in spending — is the **fiscal multiplier**.

![Pipeline showing a dollar of government spending re-spent down a chain to lift GDP](/imgs/blogs/fiscal-policy-for-traders-spending-deficits-demand-3.png)

The figure traces the chain. The government spends \$1.00 on a road contract — that is the *initial injection*. It becomes the construction company's income. Say people spend two-thirds of any extra income they receive and save the other third — economists call that fraction the **marginal propensity to consume (MPC)**, and 2/3 is a reasonable illustrative value. So the construction firm and its workers spend \$0.67 of that dollar at, say, local shops. That \$0.67 is now the shop owner's income, who spends 2/3 of *it* — \$0.44 — and so on. Each round is smaller than the last (because a fraction *leaks* out into savings each time), so the chain converges. Sum the geometric series and you get a clean formula:

```
multiplier = 1 / (1 - MPC)
```

With an MPC of 2/3, that is `1 / (1 - 2/3) = 1 / (1/3) = 3`. In a frictionless textbook world, \$1 of spending would generate \$3 of GDP. In the real world, more leaks out at each step — into savings, into taxes, into imported goods (money spent abroad does not re-circulate domestically) — so real-world multipliers are far smaller, typically somewhere between roughly 0.5 and 2 depending on conditions. A common working estimate for government purchases in a slack economy is around 1.5, which is the number we will use.

#### Worked example: the multiplier turning \$1 into \$1.50

Walk the chain explicitly with a realistic effective MPC, where leakage to savings, taxes, and imports leaves about 1/3 of each dollar re-spent domestically.

- The government spends an initial **\$1.00** on a contract. GDP so far: **\$1.00**.
- The recipient re-spends one-third domestically: **\$0.33**. Running GDP: **\$1.33**.
- The next recipient re-spends a third of that: **\$0.11**. Running GDP: **\$1.44**.
- The next round: **\$0.04**. Running GDP: **\$1.48**.
- The next: **\$0.01**. Running GDP: **\$1.49**. Subsequent rounds add fractions of a cent.

The series converges to `1 / (1 - 1/3) = 1.50`. So **\$1.00 of government spending generates about \$1.50 of GDP** once the re-spending chain plays out. That extra fifty cents is the multiplier doing its work — the same dollar counted again and again as it passes from hand to hand. Put plainly: government spending is not a one-time event but the first link in a chain of income and re-spending, and the multiplier measures how long the chain runs before it leaks away.

#### Worked example: scaling the multiplier to the CARES Act

Now apply that multiplier to a real program. The CARES Act was about **\$2.2 trillion**. Suppose a portion of it — say \$1.0 trillion of direct purchases and transfers that actually got spent — carried a multiplier of 1.5.

- Direct injection: **\$1.0 trillion**.
- With a multiplier of 1.5, the total lift to GDP: **\$1.0T × 1.5 = \$1.5 trillion**.

A \$22 trillion economy getting a \$1.5 trillion demand boost is a roughly 7% lift to GDP from that slice of the program alone — enormous, and it explains how a 2020 economy that *collapsed* in the spring came roaring back to +5.9% real growth in 2021. The multiplier is the lever arm here: a moderate-sized program, multiplied through the economy, can move aggregate demand by a sum that swamps anything a rate cut could deliver once rates are already at zero. The caveat — and it is a big one — is that the multiplier is *not* a constant; it depends entirely on the state of the economy, which is the subject of crowding out.

### What makes the multiplier big or small

Because the multiplier is the hinge of the whole subject — and because mis-stating it is the most common amateur error — it is worth spelling out what actually moves it. Four things matter most.

First, the **state of the economy**. In a deep slump with idle workers and factories, an extra dollar of demand gets *produced* (someone who was unemployed goes back to work to meet it), so the multiplier is large. At full employment, an extra dollar of demand cannot be met with more output — the economy is already producing flat-out — so it either crowds out other spending or bids up prices, and the multiplier is small. This is why the same fiscal program is powerful in 2020 and feeble in a boom.

Second, the **type of spending or tax change**. Sending a dollar to a household that is living paycheck to paycheck produces a large multiplier — they spend almost all of it (high MPC). Cutting taxes for the wealthy, who save most of a windfall, produces a small multiplier (low MPC). Direct government purchases (a road, a salary) have a multiplier of *at least* the full dollar, because the dollar is spent by definition; transfers and tax cuts depend on whether the recipient spends or saves. Policymakers who want maximum bang per deficit dollar target the spending at those most likely to spend it — which is exactly the logic of expanded unemployment benefits and stimulus checks aimed at lower-income households in 2020-21.

Third, the **monetary policy response**. If the central bank responds to fiscal expansion by *raising* rates to offset it (which it will do near full employment, to prevent overheating), the monetary tightening cancels part of the fiscal boost, shrinking the *effective* multiplier. If the central bank holds rates pinned (as at the zero bound), there is no offset and the multiplier stays large. This is a crucial subtlety: the multiplier is not a property of fiscal policy alone but of the *combined* fiscal-monetary stance. Fiscal and monetary policy pulling the same way (both easy) gives the biggest multiplier; pulling opposite ways (fiscal easy, monetary tight) gives the smallest.

Fourth, the **openness of the economy and the leakage to imports**. In an economy that imports heavily, a large fraction of every re-spent dollar leaks abroad — it becomes demand for foreign goods, supporting *foreign* GDP rather than domestic. The more open the economy, the more leakage, the smaller the domestic multiplier. This is one reason small open economies get less domestic bang from fiscal stimulus than a large, relatively closed one like the United States.

Stack those four together and you understand why credible estimates of the multiplier range so widely — from below 0.5 to above 2. The honest answer to "what is the multiplier?" is always "it depends, and here is on what." For a trader the practical upshot is simple: a fiscal program's market impact depends not on its headline dollar size but on *who gets the money, what state the economy is in, and what the Fed does in response.* A \$1 trillion program aimed at high-spenders in a slump with the Fed on hold is a vastly bigger demand event than a \$1 trillion tax cut for savers in a boom with the Fed hiking — even though the deficit number is identical.

## Crowding out: the classic objection

If government spending so reliably multiplied into bigger GDP, every government would simply spend its way to prosperity forever. It does not work that cleanly, and the reason is the most important objection in all of fiscal economics: **crowding out**.

The argument goes like this. To spend more than it taxes, the government must *borrow* — it sells bonds, which means it competes with private borrowers (companies issuing debt, households taking mortgages) for the same finite pool of savings. When a giant new borrower shows up demanding funds, the *price* of borrowing — the interest rate — gets bid up. Higher interest rates discourage private investment and consumption: a company that would have built a factory at 4% borrowing costs may shelve it at 6%. So the government's extra spending *crowds out* an equivalent amount of private spending. In the extreme version of the argument, the multiplier is roughly *zero* — every dollar the government adds, it removes from the private sector by raising the cost of capital.

This is a real mechanism and the central reason the multiplier is not a fixed number. But — and this is the part the simple version misses — **it only fully bites when the economy is at full employment**, when savings are genuinely scarce and there is no idle capacity. When the economy is in a slump, with unemployed workers, idle factories, and households and businesses *hoarding* savings rather than investing, the picture inverts.

![Two columns contrasting crowding out at full employment with crowding in during a slump](/imgs/blogs/fiscal-policy-for-traders-spending-deficits-demand-5.png)

The figure puts the two regimes side by side, both starting from the *same* bond sale. On the left, **crowding out** in a full-employment economy: the Treasury sells \$1,000,000,000 of new bonds into a pool of savings that is already fully deployed. The government is now bidding against private firms for scarce funds, so yields nudge up, and higher yields crowd out private investment. On the right, **crowding in** during a slack, recessionary economy: the Treasury sells the *same* \$1,000,000,000, but now the savings are idle and private demand is already weak — there is nobody to compete with. Yields can stay low. And the deficit-funded demand can actually *pull in* private activity that the weak economy had pushed to the sidelines, the opposite of crowding out.

The 2020 evidence is dramatic and decisive on this point. The US ran the largest deficit in its peacetime history — \$3.13 trillion — and sold a corresponding tidal wave of bonds. If crowding out were automatic, yields should have spiked. Instead, the 10-year Treasury yield was around **0.93% at the end of 2020**, near record lows, and had touched **0.62%** that summer. The biggest bond supply in history coincided with some of the *lowest* yields in history. Why? Because the economy was deeply slack (unemployment had spiked to 14.8% in April 2020), savings were piling up unspent, and — not incidentally — the Fed was buying enormous quantities of those bonds through QE. The savings competition that drives crowding out simply was not binding.

#### Worked example: a \$1 billion bond issue and the yield it nudges

Make the crowding-out mechanism concrete with a single auction. Suppose the Treasury needs to sell **\$1,000,000,000** of new 10-year notes to fund part of the deficit, and compare two states of the world.

- **Full-employment world.** The pool of available savings is, say, effectively fixed at the current yield. To attract an extra \$1,000,000,000 of buyers, the Treasury must offer a slightly higher yield — say it has to lift the auction yield by **0.05%** (5 basis points) to clear. Every private borrower issuing debt that week now faces that same 0.05% higher benchmark. On the roughly \$50 trillion of US bond-market debt that reprices off Treasuries over time, even a few basis points is a meaningful increase in the economy's cost of capital — and at the margin, some private investment that pencilled at the old yield no longer does. That marginal private spending is what got crowded out.
- **Slack world (2020).** The same **\$1,000,000,000** sale meets idle savings and a Fed standing in the market as a buyer. The yield needed to clear the auction does *not* rise — in 2020 yields actually *fell* through the largest issuance in history. Nothing is crowded out, because nothing was competing for the funds.

The lesson in dollars: the *same* \$1,000,000,000 bond sale can lift yields and crowd out private investment in a hot economy, or clear at unchanged-to-lower yields and crowd nothing out in a cold one. Crowding out is fundamentally a story about *scarcity of savings* — it bites hard when capital is fully employed and barely at all when capital is sitting idle.

#### Worked example: fiscal versus monetary scale in 2020

This is the comparison that defines the whole post. Put the two 2020 responses on the same ruler.

- **The monetary response.** The Fed cut its policy rate from a target upper bound of **1.75%** to **0.25%** — a total of **1.50 percentage points** of cuts — and then it was *done*, pinned at the zero bound with no conventional room left. It also expanded its balance sheet by trillions through QE, but QE swaps reserves for bonds; it does not put spendable money into household checking accounts. Its direct contribution to *spendable household income*: roughly **\$0**.
- **The fiscal response.** Congress ran a **\$3.13 trillion** deficit — money that went *directly* into household accounts, business payrolls, and the broader economy as spendable demand. With even a conservative multiplier, that translates into multiple trillions of GDP impact.

Set them side by side: **a 1.5-percentage-point rate cut that hit a wall, versus \$3.13 trillion of direct spendable demand.** There is no contest. Once the policy rate is at zero, the monetary lever is out of room and the fiscal lever is the only one that can move the needle. Seen plainly: monetary policy is a multiplier on private borrowing — and at the zero bound that multiplier is jammed — whereas fiscal policy is demand injected straight into the economy, which is exactly why 2020-21 was a fiscal episode wearing a monetary costume.

## When fiscal dominates monetary: the zero bound

We have circled the central thesis several times; now let us state it cleanly. **When monetary policy is at the zero lower bound, fiscal policy becomes the dominant macro force.** Understanding *why* turns this from a slogan into a tool.

The **zero lower bound (ZLB)** is the floor under the policy rate. A central bank can cut its rate to roughly zero, but it cannot cut much below — if it tried to charge meaningfully negative rates, people would simply hold physical cash, which yields zero, rather than accept a guaranteed loss. So at the ZLB, the conventional monetary lever is *exhausted*: it cannot stimulate further by cutting, because it has nothing left to cut. (For the full mechanics of how the central bank sets and is constrained by this rate, see [How the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates) and the companion piece on the [central bank toolkit of rates, QE, QT, and forward guidance](/blog/trading/macro-trading/central-bank-toolkit-rates-qe-qt-forward-guidance).)

At the ZLB, two things change the fiscal-monetary balance of power. First, the *direct* nature of fiscal policy becomes decisive: it does not need lower interest rates to work, so the zero bound does not constrain it at all. The government can spend regardless of where rates are. Second, crowding out essentially *disappears*. At the ZLB the economy is, by construction, deeply slack — that is *why* rates got cut to zero — so the savings competition is not binding, and the multiplier is at its *largest*. Fiscal policy is most powerful precisely when monetary policy is most powerless. They are complements in a crisis: the central bank pins rates at the floor and buys the bonds, which keeps fiscal borrowing cheap; the Treasury does the actual demand injection.

There is an even deeper version of this idea. When the central bank holds rates at zero *and* commits to keeping them there while the government floods the economy with deficit spending, you can get into what economists call **fiscal dominance** — a regime where fiscal needs effectively dictate monetary policy, where the central bank cannot raise rates without bankrupting the government's interest bill, and where the deficit, not the central bank, sets the inflation path. We are not fully in that regime in the US, but the rising **net interest** on the debt — federal interest outlays nearly tripled from about \$345 billion in 2020 to roughly \$1 trillion by 2025, and in fiscal 2024 interest exceeded the entire national defense budget for the first time — is exactly the pressure that *creates* fiscal dominance. As the debt and the deficit grow, the fiscal tail increasingly wags the monetary dog. That is one of the most important slow-moving macro stories of the decade, and it is fundamentally a *fiscal* story.

For the broader money-and-liquidity backdrop here — base money, broad money, and how the two relate — the companion piece [What money really is](/blog/trading/macro-trading/what-money-really-is-base-money-broad-money-traders) builds that foundation; fiscal policy operates on top of it, moving demand while monetary policy moves the money.

### Net interest: the deficit that feeds on itself

There is a feedback loop inside the fiscal arithmetic that is becoming one of the most important forces in markets, and it deserves its own moment. When the government runs deficits, the debt grows. When the debt grows, the *interest* the government pays on that debt grows. And interest payments are themselves *spending* — so a rising interest bill widens the deficit further, which grows the debt further, which raises interest again. It is a loop that can feed on itself.

The numbers have turned sharply. Federal **net interest** outlays — the cost of servicing the debt — ran about \$345 billion in 2020, when rates were near zero. By 2023 they had jumped to roughly \$658 billion, and by 2024-25 they crossed roughly \$880 billion toward \$1 trillion a year. In fiscal 2024, for the first time in modern history, the government spent more on *interest on the debt* than on *national defense*. Two forces drove that: the debt stock kept growing (more principal to pay interest on), and the Fed's hiking cycle pushed the *rate* on new and refinanced debt up from near zero to 4-5%. As the old, cheap debt matures and gets refinanced at today's higher yields, the average interest rate on the whole stock keeps climbing, and the interest bill keeps rising even if the debt stopped growing.

Why does this matter for trading? Because it is the mechanism that creates **fiscal dominance** — the regime where the central bank's hands are partly tied by the government's interest bill. When net interest is small, the Fed can raise rates freely to fight inflation. When net interest is a trillion dollars and climbing, every rate hike directly worsens the deficit, and the political and economic pressure to *stop* hiking — or to tolerate higher inflation that erodes the real value of the debt — grows. A trader watching the long end of the curve in the mid-2020s is, in part, watching the market price the *odds* of fiscal dominance: the rising **term premium** (the extra yield investors demand to hold long bonds) partly reflects the fear that relentless deficits and a swelling interest bill will eventually be resolved through higher inflation rather than spending cuts. That is a fiscal story driving a bond-market price, and it is invisible to anyone watching only the Fed.

## Common misconceptions

A few myths cause more bad fiscal trades than anything else. Each gets corrected with a number.

**Myth 1: "Deficits always crowd out private investment and lift yields."** This is the most common error, and 2020 demolishes it. The US ran a **\$3.13 trillion** deficit — the largest peacetime issuance in its history — and the 10-year yield *fell* to **0.93%**, having touched **0.62%** that summer. Crowding out is conditional on a full-employment economy with scarce savings. In a slump with idle capacity and a central bank buying bonds, a record deficit can coincide with record-low yields. The deficit's effect on yields depends entirely on the state of the cycle — never assume it is automatic.

**Myth 2: "Government spending is always inflationary."** Also false, and again the state of the economy is everything. In 2020, with the output gap enormous (the economy producing far below capacity, unemployment at 14.8%), the giant fiscal injection mostly *refilled* lost demand without much inflation — CPI was actually *near zero* in mid-2020 (0.12% year-over-year in May 2020). The inflation came *later*, in 2021-22, once the demand surge collided with supply constraints and the economy hit its capacity — CPI peaked at **9.06% in June 2022**. Spending into a slack economy fills a hole; spending into a full one overflows it. The same fiscal dollar is disinflationary in one regime and inflationary in another. (For more on how that inflation eventually forced the Fed's hand, see the companion on the [central bank toolkit](/blog/trading/macro-trading/central-bank-toolkit-rates-qe-qt-forward-guidance).)

**Myth 3: "The fiscal multiplier is always greater than 1."** No — the multiplier is a *range*, not a constant, and it can be well below 1. In a full-employment economy with crowding out, the multiplier on government purchases can be **0.5 or lower**, because the deficit displaces private spending and imports leak demand abroad. The multiplier is large (perhaps 1.5 or more) only in a deep slump with idle resources and rates at the floor — exactly 2020's conditions. Quoting "the multiplier is 1.5" as a universal constant is the single most common modelling mistake amateurs make. Always ask: *what regime are we in?*

**Myth 4: "The deficit and the national debt are the same thing."** They differ by more than a factor of ten. The **deficit** is the annual flow (\$3.13 trillion in 2020, roughly \$1.8 trillion in 2025); the **debt** is the cumulative stock of all past deficits (past \$37 trillion in 2025). A "shrinking deficit" still *adds* to the debt — it just adds less. The debt only falls when the budget actually runs a surplus, which has not happened since 2001.

**Myth 5: "A government can run out of money like a household."** A government that borrows in *its own currency* (like the US) cannot be forced into default by an inability to pay — it can always create the currency to meet a payment. The real constraints are *inflation* (if it creates too much) and *the bond market's willingness to lend at a reasonable rate*. This is why the binding fiscal signal for traders is not "can they pay?" but "what does the new supply do to yields and to inflation?" — which is exactly what the playbook tracks.

## How it shows up in real markets

Theory is one thing; let us walk the actual 2020-21 episode as it printed on screens, because it is the cleanest fiscal case study in living memory.

### The 2020 demand surge

The sequence ran like this. The economy locked down in March 2020; aggregate demand collapsed as `C` and `I` cratered; real GDP fell **−2.2%** for the year (and far more sharply in Q2). The fiscal response was immediate and gigantic — CARES Act in late March, more relief in December, the American Rescue Plan in March 2021 — driving the deficit to **15.0% of GDP** in 2020 and **12.3%** in 2021. That fiscal injection, landing on a household sector that had been forcibly prevented from spending, built up an enormous stock of savings and pent-up demand. When the economy reopened, that demand came flooding out: real GDP *rebounded to +5.9% in 2021*, the fastest growth in decades.

![Bars of the deficit percent of GDP overlaid with a line of real GDP growth, 2019 to 2025](/imgs/blogs/fiscal-policy-for-traders-spending-deficits-demand-6.png)

The figure makes the fiscal-demand link visible. The amber bars are the deficit as a percent of GDP; the blue line is real GDP growth. Watch them move together: the deficit explodes to 15% in 2020 exactly as growth craters to −2.2% (fiscal policy catching a falling economy), and then the still-huge 12.3% deficit in 2021 coincides with the +5.9% growth rebound (the deficit-fuelled demand surge). This co-movement is the empirical signature of fiscal policy driving aggregate demand. The deficit is not a passive consequence of the cycle here — it is an active driver of it. A trader watching this in real time, who understood that a 15%-of-GDP fiscal impulse would force a demand rebound once the economy reopened, had a powerful, early thesis for the 2021 reflation: long cyclicals, long commodities, short duration.

### The inflation that followed

Then came the part that taught a generation of traders to respect fiscal policy. The demand surge eventually outran the economy's capacity to produce, and — colliding with snarled supply chains — it ignited the worst inflation in forty years. CPI ran near zero in mid-2020 (the slack absorbing the spending), then climbed steadily through 2021 as demand recovered, and peaked at **9.06% in June 2022**. Core PCE, the Fed's preferred gauge, peaked at **5.6%** in early 2022 against a 2% target.

The causal chain is worth stating precisely because it is the heart of the fiscal-trading lesson: an extraordinary *fiscal* impulse drove an extraordinary demand surge, which — once the slack was used up — drove an extraordinary *inflation*, which forced an extraordinary *monetary* response (the Fed hiking from 0.25% to 5.50% over 2022-23). The whole 2020-2023 macro cycle — the crash, the boom, the inflation, the hiking cycle — was *initiated by fiscal policy*. A trader who watched only the Fed and ignored the deficit saw the second half of the movie and missed the cause. The deficit blowout in 2020-21 was the leading indicator for the inflation of 2022 and the rate hikes that followed.

That is the case for never trading macro with fiscal policy switched off. It does not just matter at the margin; in the defining macro episode of the decade, it was the *first cause*.

### The debt ceiling: when fiscal politics becomes a market event

There is one more way fiscal policy reaches your screen that is pure politics: the **debt ceiling**. The US has a statutory cap on how much the Treasury can borrow, and periodically Congress has to vote to raise it. When that vote gets stuck — as it did in 2011, 2013, and 2023 — the Treasury cannot issue net new debt, which forces a strange and tradable sequence of events. To keep paying the bills without borrowing, the Treasury drains its cash account (the Treasury General Account) and deploys "extraordinary measures." This temporarily *adds* liquidity to the financial system (the Treasury is spending down its cash rather than pulling cash in through borrowing), which can be a quiet tailwind for risk assets. Then, once the ceiling is finally raised, the Treasury *rebuilds* its cash by issuing a flood of bills all at once — draining liquidity back out, often abruptly, which can pressure risk assets and lift short-term funding rates. The 2023 episode saw the Treasury's cash balance fall to about \$0.05 trillion at the June low before a post-resolution issuance surge rebuilt it. A trader who understands the fiscal plumbing positions *around* these events: the drain-then-flood pattern of the cash account is a recurring, datable, fiscal-driven liquidity swing that has nothing to do with the Fed's policy rate and everything to do with the Treasury's borrowing calendar.

## How to trade it: the playbook

Everything above resolves into a small set of things to track and a small set of positions they imply. Here is the working playbook.

![Grid mapping three fiscal signals to what they tell you and how to trade each](/imgs/blogs/fiscal-policy-for-traders-spending-deficits-demand-7.png)

The figure is the playbook in one frame: three fiscal signals down the left, what each one tells you in the middle, and the concrete trade on the right. Work through them.

**Signal 1 — track the fiscal impulse (the *change* in the deficit).** The level of the deficit is old news; markets care about the *direction and pace of change*. A *widening* deficit (positive impulse) is fresh demand being added — pro-growth, reflationary, more bond supply coming. A *narrowing* deficit (negative impulse — "fiscal drag") is demand being withdrawn — a growth headwind, disinflationary, less supply ahead. Concretely: when a large stimulus is being debated and its odds of passing rise, the impulse is turning positive, and the trade leans pro-cyclical — long equities and cyclical sectors, long industrial commodities (copper, oil), and *short* duration (sell bonds), because more demand and more supply both push yields up. When a stimulus is *rolling off* (a "fiscal cliff") or austerity is being legislated, flip it: that fiscal drag argues for *lower* yields and a more defensive equity stance. The impulse, not the level, is the tradable signal.

**Signal 2 — watch the deficit trajectory against the ~3% stabilizing line.** A deficit persistently *above* 3% of GDP and *rising* means the debt ratio is climbing, bond supply is heavy and growing, and there is structural upward pressure on both inflation risk and the term premium. The US has been running 6%-ish deficits in a *full-employment* economy in the mid-2020s — which is historically unusual and exactly the kind of late-cycle fiscal expansion that *can* crowd out and lift the long end. The trade that this trajectory has favoured is a *steeper* curve (the long end selling off relative to the front) and a rebuilt term premium — short the long bond, or position for curve steepeners — because the market has to be paid more to absorb a relentless flood of new duration. (The mechanics of how that issuance hits the curve are the subject of the forthcoming companion on [deficits, debt, and why issuance moves yields](/blog/trading/macro-trading/deficits-debt-bond-supply-why-issuance-moves-yields).)

**Signal 3 — track the issuance the deficit forces.** Every dollar of deficit is a bond that has to be sold, so the deficit trajectory *is* the bond-supply forecast. Watch the Treasury's quarterly **refunding** announcements (which tell you how much of which maturities they will issue), and watch the *auction internals* — the **bid-to-cover ratio** (demand relative to supply) and the **tail** (whether the auction cleared at a higher yield than expected). A string of weak auctions — low bid-to-cover, big tails — into a heavy-supply quarter is a real-time signal that the market is choking on supply and that yields, especially at the long end, are biased higher. This is where the abstract deficit becomes a concrete, datable, tradable flow on specific calendar days.

**Putting it together — the position and the invalidation.** The core fiscal trade has a clean structure. *Thesis:* a large positive fiscal impulse into a slack economy is pro-growth, reflationary, and supply-heavy — so lean long risk and short duration. *Invalidation:* the impulse turns negative (deficit starts narrowing faster than expected — a fiscal cliff, a debt-ceiling spending freeze, or legislated austerity), **or** the economy hits full employment so that further fiscal expansion crowds out and tips from reflationary into stagflationary. Both of those flip the trade. The single discipline that ties the whole playbook together: **never read the Fed without also reading the Treasury.** Monetary policy sets the price of money, but fiscal policy moves the demand and supplies the bonds — and when the Fed is out of room, fiscal policy *is* the macro. Track the impulse, the trajectory, and the issuance, and you are reading the half of macro that most traders leave switched off.

## Further reading and cross-links

Within this series, the natural companions are:

- [What money really is: base money, broad money, and what traders need](/blog/trading/macro-trading/what-money-really-is-base-money-broad-money-traders) — the money foundation that fiscal policy operates on top of.
- [The central bank toolkit: rates, QE, QT, and forward guidance](/blog/trading/macro-trading/central-bank-toolkit-rates-qe-qt-forward-guidance) — the *monetary* half of macro, and how it complements fiscal policy at the zero bound.
- [Deficits, debt, and bond supply: why issuance moves yields](/blog/trading/macro-trading/deficits-debt-bond-supply-why-issuance-moves-yields) — the deep dive on how the deficit turns into bond supply and how that supply hits the curve.
- [How the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates) — the rate-setting mechanics and the zero lower bound that makes fiscal policy dominant in a crisis.

The thread through all of them is the same: macro has two engines, monetary and fiscal, and the deficit is the bolt that joins them. Read both engines, watch the bolt, and you are reading the whole machine.
