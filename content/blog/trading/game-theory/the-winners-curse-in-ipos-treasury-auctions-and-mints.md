---
title: "The Winner's Curse in IPOs, Treasury Auctions, and Mints"
date: "2026-06-18"
publishDate: "2026-06-18"
description: "Winning a common-value auction means you were the most optimistic bidder, so you must shade your bid down for the curse and shade more as the field grows."
tags: ["game-theory", "winners-curse", "auctions", "ipo", "treasury-auctions", "nft-mints", "common-value", "bid-shading", "trading"]
category: "trading"
subcategory: "Game Theory"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — In any auction where everyone is bidding for the *same* underlying worth and each of you has a noisy guess of it, the simple act of winning tells you something brutal: you were the most optimistic estimator in the room, so your guess was probably too high — and that is the winner's curse.
>
> - **Winning is information.** If you won a common-value auction by bidding your honest estimate, you almost certainly overpaid, because the winner is selected for having the highest (most over-optimistic) signal.
> - **Shade your bid down — and shade *more* as the field grows.** With 2 rivals on a \$100 item with \$20 of estimate noise you shave about \$6.67; with 20 rivals you shave about \$18.10. More competition makes the curse *worse*, not better.
> - **The three arenas:** IPOs (a full allocation is bad news — you only get filled on deals the informed passed on), Treasury auctions (uniform-price softens the curse; pay-your-bid sharpens it), and NFT/token mints (hyped mints curse the winners hardest because the noise is enormous).
> - **The one rule:** estimate the value, then *subtract* the curse — the expected overpayment of the winner, which rises with both the number of bidders and the noise in your estimate. Bid that number, not your estimate.

In May 2012, Facebook went public at \$38 a share. The deal was the most hyped IPO in a generation; retail investors begged their brokers for allocation, and many got exactly what they asked for. Within three months the stock traded near \$18 — down more than 50%. The people who got their *full* allocation were not the lucky ones. They were the ones nobody fought them for.

That is the winner's curse, and it is one of the most important — and most counterintuitive — ideas a trader can carry. We met it once already in this series, in the context of order flow: in [the adverse-selection post](/blog/trading/game-theory/adverse-selection-and-the-winners-curse-why-a-fast-fill-is-bad-news) we saw that a suspiciously *fast fill* on a resting order is bad news, because the only person eager to take the other side of your trade right now is someone who knows something you don't. This post takes that same idea and points it at the explicit, sealed-bid **auctions** a trader actually competes in: the IPO bookbuild, the Treasury bill auction, the NFT mint, the token sale. In all of them, the prize is the same underlying worth, everyone has a noisy estimate of it, and *winning means you were the most optimistic estimator*. Unless you saw that coming and shaded your bid down, you overpaid.

![Winning a common-value auction means you were the most optimistic bidder; the naive winning bid sits above the true value and the gap widens as more bidders pile in](/imgs/blogs/the-winners-curse-in-ipos-treasury-auctions-and-mints-1.png)

The chart above is the whole post in one model. The dashed line is the item's true value — \$100. The red line is the bid the *naive* winner submits, and it sits stubbornly above \$100, pulling further away as the field gets larger. That gap is money the winner hands over for nothing. By the end of this post you will know how to compute that gap, why it grows with competition, and how to bid so you keep it instead of paying it.

## Foundations: common-value auctions, the curse, and how to shade

Before any arithmetic, we need to build the vocabulary from zero. An *auction* is just a mechanism for deciding who gets a thing and at what price, when several people want it and the seller will not simply name a number. There are many flavors, but the single distinction that drives the winner's curse is **what the thing is worth to the bidders**.

### Private-value vs common-value auctions

In a **private-value auction**, the item is worth a different amount to each bidder, and each bidder knows their own number for sure. The classic example is a painting at an estate sale: I might value it at \$500 because it matches my living room, you might value it at \$2,000 because it reminds you of your grandmother, and neither of us is wrong. There is no single "true" value we are both trying to estimate. In a private-value world the winner's curse barely exists — if I bid \$480 and win, I got a thing I genuinely valued at \$500 for \$480, and I am happy.

In a **common-value auction**, the item is worth the *same* amount to everyone — there is one true number — but *nobody knows it exactly*. Each bidder only sees a noisy private estimate of it. The textbook example is an oil lease: the barrels of crude under the ground are worth the same dollar amount no matter who pumps them, but each oil company's geologists produce a slightly different estimate of how much is down there. The number is common; the *signals* are private and noisy.

Financial auctions are overwhelmingly **common-value**. A 10-year Treasury note auctioned today has one correct price — the present value of its coupons and principal, given today's interest rates. Every primary dealer is trying to estimate that one number. A newly public company has one true intrinsic value; every fund bidding in the bookbuild is estimating it. A freshly minted NFT collection has one eventual floor price; every minter is guessing at it. This single fact — *one true value, many noisy guesses* — is the engine of everything that follows.

> [!note]
> A *signal* is just your private piece of evidence about the true value — your analyst's price target, your geologist's reserve estimate, your read of the hype. It equals the true value plus an error. The error is what makes you sometimes too high and sometimes too low.

### Why winning is bad news: the selection effect

Here is the mechanism, stated as plainly as possible. Suppose the true value is \$100 and there are ten bidders, each of whom sees an estimate that is the true value plus a random error — sometimes \$90, sometimes \$110, scattered around \$100. If everyone naively bids their estimate, who wins? *The bidder who drew the highest estimate.* And the highest estimate out of ten is, by construction, an over-estimate. The winner is not a random bidder; the winner is **selected for being the most over-optimistic**. So conditional on winning, your estimate was biased high, and your bid — which equalled your estimate — was too high. You won the auction and lost money. That is the curse.

The crucial, counterintuitive twist is the effect of competition. Your instinct says: *more bidders means I have to bid more aggressively to win.* That instinct is exactly backwards. More bidders means the *highest of all those estimates* is even more extreme — the maximum of twenty draws is further above the mean than the maximum of two — so the curse is *worse*. The correct response to a crowded auction is to bid *less* aggressively, not more. We will quantify this in a moment, and it is the single most valuable thing in this post.

There is a famous classroom demonstration of all of this — the **jar of coins**. A professor fills a glass jar with coins, holds it up, and auctions it to the class; whoever bids highest pays their bid and keeps the jar. The jar has one true value (the coins are worth exactly what they're worth), but every student only *guesses* it, and the guesses scatter widely because counting coins through glass is hard. Run this experiment and two things happen with iron regularity. First, the *average* guess is usually close to the true value — the class as a whole is wise. Second, the *winning* guess is wildly too high, and the winner loses money almost every single time. The class is collectively smart and the winner is individually cursed, and those two facts are the same fact: the winner is the one tail of an otherwise sensible distribution. The jar is not a trick about coins; it is the structure of every common-value auction, and the IPO bookbuild, the Treasury auction, and the NFT mint are all jars of coins held up by people who'd like you to bid high.

The reason this matters so much is that the curse is *invisible from the inside*. When you win, you don't feel cursed — you feel like you got the thing you wanted, at a price you were willing to pay. The bad news is silent and statistical: it shows up only across many auctions, as a slow drag on your returns that you can easily mistake for bad luck. A trader who doesn't model the curse will lose to it for years while attributing the losses to everything else. That is exactly why we put numbers on it.

### Bid shading: the fix

The fix is called **bid shading**: you do not bid your estimate; you bid your estimate *minus* a discount that accounts for the curse. The size of the discount — the *shave* — is exactly the expected amount by which the winner over-estimates. Shade by that amount and, on average, when you win you break even instead of overpaying. Shade by more and you win less often but make money when you do; shade by less and you win the "hot" auctions and bleed.

Formally, our model uses a clean, computable version of this. Suppose each bidder's signal is the true value $v$ plus an error drawn uniformly between $-w$ and $+w$, where $w$ is the *noise* — how uncertain your estimate is. Then the expected error of the *winning* (highest) signal out of $n$ bidders is

$$\text{expected overpayment} = w \cdot \frac{n-1}{n+1}.$$

Here $w$ is the half-width of your estimate error (in dollars), $n$ is the number of bidders, and the result is how far above the true value the naive winning bid sits. That is the entire winner's-curse model, and every number in this post comes straight out of it (via `data_gametheory.winners_curse`). Notice the two levers: the overpayment rises with the noise $w$ and rises with the field size $n$. Both make the curse worse.

### Auction formats: the rules that change the game

One more piece of foundation. The *format* of the auction — the rule for who pays what — changes how hard the curse bites, so we need the vocabulary:

- **Sealed-bid first-price (pay-your-bid / discriminatory):** everyone submits a sealed bid; winners pay exactly what they bid. This is the harshest format for the curse, because overbidding by even a dollar costs you that dollar directly.
- **Uniform-price (single-price):** everyone submits a sealed bid, but every winner pays the *same* clearing price — the lowest accepted bid. A slightly-too-high bid still only pays the market price, so the penalty for eagerness is muted.
- **Ascending (English) auction:** the price rises and bidders drop out; you see others quit, which leaks information and partly cures the curse. This is the eBay/art-auction format.
- **Descending (Dutch) auction:** the price *falls* from a high start until someone accepts; the first to accept wins at that price. Used in some IPOs and many NFT mints.

Keep these four straight — we will meet uniform-price and discriminatory in the Treasury section and Dutch in the mints section. Now let us put real numbers on the curse.

## Putting numbers on the curse

The model says the expected overpayment of the naive winner is $w \cdot \frac{n-1}{n+1}$. Let us walk it, because the arithmetic is where the intuition becomes a tool you can actually bid with.

#### Worked example: the curse with ten bidders

You are bidding on an item you believe is worth \$100. Your estimate carries noise of $w = \$20$ — meaning your private signal could be off by up to \$20 in either direction. There are $n = 10$ bidders, all in the same boat. Plug into the model:

$$\text{overpayment} = 20 \cdot \frac{10-1}{10+1} = 20 \cdot \frac{9}{11} = \$16.36.$$

So the naive winner — whoever bids their honest \$100-ish estimate and happens to have drawn the highest one — pays, on average, \$116.36 for a thing worth \$100. They overpay by \$16.36, which is more than 16% of the item's value, purely from the selection effect of having won. To break even, you must bid your estimate *minus* \$16.36 — that is, around \$83.64. The intuition: **in a crowded auction, winning at your honest estimate is a near-guarantee that you overpaid, and the cure is to bid as if your estimate were \$16 too high.**

That is `data_gametheory.winners_curse(true_value=100, n_bidders=10, signal_noise=20)`, and it returns an expected overpayment of \$16.36 — the number on the cover chart.

#### Worked example: how the shave grows with the field

Now hold the item (\$100) and the noise (\$20) fixed, and vary only the number of bidders. The required shave — the amount you subtract from your estimate to break even — is exactly the overpayment, so:

| Bidders $n$ | $\frac{n-1}{n+1}$ | Required shave | Disciplined bid |
|---|---|---|---|
| 2 | 1/3 | \$6.67 | \$93.33 |
| 5 | 4/6 | \$13.33 | \$86.67 |
| 10 | 9/11 | \$16.36 | \$83.64 |
| 20 | 19/21 | \$18.10 | \$81.90 |
| 50 | 49/51 | \$19.22 | \$80.78 |

Read down the table. Going from a sleepy 2-bidder auction to a crowded 20-bidder one, the shave nearly *triples*, from \$6.67 to \$18.10. The disciplined bid falls from \$93.33 to \$81.90. The instinct to "bid up because there's more competition" would have you doing the exact opposite of what the model demands. The intuition: **each additional rival you must beat means the winner's estimate is more extreme, so you must bid more cautiously, not less.**

![The more bidders you beat, the harder you must shave your bid; the required shave rises from about six dollars with two rivals toward twenty dollars with fifty](/imgs/blogs/the-winners-curse-in-ipos-treasury-auctions-and-mints-2.png)

The amber curve above is that shave, drawn for every field size from 1 to 20. Notice its *shape*: it rises steeply at first — going from 2 to 5 bidders adds almost \$7 of required shave — then flattens. The factor $\frac{n-1}{n+1}$ approaches 1 as $n$ grows, so the shave approaches the full noise $w = \$20$ but never quite reaches it. There is a ceiling to the curse: even against a thousand bidders you never shade by more than the full width of your own uncertainty. But you get most of the way there by the time the field hits a few dozen.

### The curse scales with your ignorance

The other lever is the noise $w$. The more uncertain your estimate, the bigger the curse — and this is the lever that separates a boring, well-understood asset from a wild one.

#### Worked example: noise is the curse's accelerator

Fix the field at $n = 10$ bidders and vary the noise. The overpayment is $w \cdot \frac{9}{11} = 0.818 \, w$, so it is simply *proportional* to how uncertain you are:

| Noise $w$ | Overpayment ($0.818 \, w$) |
|---|---|
| \$5 | \$4.09 |
| \$10 | \$8.18 |
| \$20 | \$16.36 |
| \$40 | \$32.73 |

A well-understood, liquid asset — a 4-week Treasury bill, say, whose fair price everyone can compute to the penny — has tiny noise, so the curse is tiny and bidders can bid near fair value. A hyped early-stage IPO or a brand-new NFT collection has *enormous* noise: nobody knows what it's worth, estimates are scattered \$40 wide, and the curse balloons to \$33 on a \$100 item. The intuition: **the curse is not just about how many people you're bidding against — it's about how little anyone actually knows, and the least-knowable assets carry the most dangerous curse.**

![The noisier your estimate, the bigger the curse you must price in; expected overpayment grows in proportion to the noise, reaching over thirty dollars at forty dollars of noise](/imgs/blogs/the-winners-curse-in-ipos-treasury-auctions-and-mints-4.png)

The red line climbs in a straight line from the origin: double the noise, double the overpayment. This is why the same auction *format* can be benign for one asset and ruinous for another. The format sets the field; the asset sets the noise; the curse is the product of the two.

## Arena 1: IPOs and the adverse-selected allocation

An **IPO** (initial public offering) is the first time a private company sells shares to the public. Most large IPOs are not run as a literal sealed-bid auction; they use a **bookbuild**, where an investment bank (the *underwriter*) takes indications of interest from institutional investors, gauges demand, sets a price, and *allocates* shares. But the winner's-curse logic governs it completely — just dressed in different clothes.

### Why a full allocation is bad news

Here is the structure. There is one true value for the new company's shares, and every fund is estimating it with noise. The bank can only allocate the fixed number of shares on offer. When a deal is *hot* — when the informed, well-connected funds have done the work and concluded it's underpriced — demand massively exceeds supply. The book is **oversubscribed**, often 5x or 10x. The bank then *rations*: everyone who asked for shares gets only a fraction.

When a deal is *cold* — when the smart money has looked and quietly walked away — demand falls short of supply. The book is **undersubscribed**. There is no need to ration anyone. You ask for your full size, and you *get* your full size.

Now stitch those two together from the point of view of an uninformed bidder who asks for the same dollar amount in every deal. On the good deals, you are rationed down to a sliver. On the bad deals, you are filled in full. Averaged over many IPOs, *your filled book is dominated by the deals the informed money rejected*. A full allocation is not a victory; it is the market telling you the people who knew better passed. This is exactly the adverse-selection structure from the order-flow post — you are *selected into* the deals you should have avoided — and economists call it the **Rock model** of IPO underpricing, after Kevin Rock's 1986 paper.

![IPO allocation is adverse-selected: hot deals are rationed so you get a sliver while cold deals are filled in full so you get everything, leaving your book skewed toward the deals informed buyers avoided](/imgs/blogs/the-winners-curse-in-ipos-treasury-auctions-and-mints-3.png)

The before/after above shows the two cases side by side. Read the left column (the hot deal): demand is 10x the shares, you ask for 1,000 and the bank hands you 100 — a tiny fill on the deal that pops. Read the right column (the cold deal): demand is below the book, no rationing is needed, and you receive all 1,000 — a full fill on the deal that sags. Same request both times; opposite outcomes; and the outcome you "win" is the bad one.

#### Worked example: the uninformed IPO bidder's real return

Suppose you bid in two IPOs, asking for \$10,000 of stock in each. One is hot and pops 30% on day one; one is cold and falls 20%. If you got equal fills, your average return would be a healthy $(30\% - 20\%)/2 = +5\%$. But you don't get equal fills.

The hot deal is oversubscribed 10x, so you receive only \$1,000 of the \$10,000 you asked for. The cold deal is undersubscribed, so you receive the full \$10,000. Now compute your actual P&L:

- Hot deal: \$1,000 at +30% = +\$300.
- Cold deal: \$10,000 at −20% = −\$2,000.
- Total: \$300 − \$2,000 = **−\$1,700** on \$11,000 deployed, a return of about **−15.5%**.

The naive average said +5%; the *allocation-weighted* reality is −15.5%. The intuition: **the rationing on the deals you wanted, combined with the full fills on the deals you didn't, is the winner's curse wearing an IPO costume — and it is why the table is tilted against the uninformed bidder before the stock even trades.**

### IPO underpricing as the compensation

So why does anyone uninformed play at all? Because the market *compensates* them — through **IPO underpricing**. On average, IPOs are priced below where they will trade, leaving a first-day "pop." Across decades of US IPOs the average first-day return has been roughly **+15% to +18%** (the long-run average in Jay Ritter's widely cited data is around 18%, with huge year-to-year swings — over 70% in the 1999 dot-com frenzy, near zero or negative in cold years). That pop is not a free lunch; it is the *premium the issuer leaves on the table to keep uninformed bidders in the game* despite the adverse-selected allocation. Without it, the uninformed would compute the −15.5% above, quit, and the bank would lose the demand it needs to get deals done. Underpricing is the bribe that exactly offsets the curse — in equilibrium, just barely.

The lesson for a trader: the first-day pop is real but it is *conditional on getting filled*, and you get filled least on the deals that pop most. The headline "IPOs pop 18% on average" is a statement about shares offered, not about the shares *you* will actually hold. Your realized return is dragged toward the cold deals you got stuffed with.

### The bookbuild: where the information is extracted

It's worth understanding *why* IPOs use a bookbuild rather than a clean auction, because the answer reveals the game the bank is really playing. When the underwriter goes on the roadshow, it isn't just marketing — it's *extracting information*. The bank asks big institutions for their indications of interest at various prices, and those indications are *signals* about the true value. A fund that says "I want 500,000 shares at \$20 but nothing at \$24" has just leaked its private estimate. The bank aggregates all those signals to set the final price — it is, in effect, running a noisy poll of the informed crowd.

But here's the catch that ties it back to the curse: a fund that reveals strong demand is helping the bank price the deal *higher*, which is against the fund's own interest. So the bank has to *reward* honest revelation, and it does so with — you guessed it — allocation and underpricing. Funds that show up early with strong, honest demand on the *good* deals get preferential allocation, and they get it at a price that leaves them a pop. This is the **information-extraction theory** of IPO underpricing (Benveniste and Spindt, 1989): underpricing is the fee the issuer pays the informed institutions to tell the truth about value. The uninformed retail bidder is not part of this conversation; they get the leftover allocation and the curse. The bookbuild is a machine for paying informed bidders to reveal their signals, and the uninformed are the ones who fund the payment.

#### Worked example: the informed vs the uninformed in one deal

Consider a deal where the true value is \$22 but the bank prices it at \$20 to leave a pop. There are two kinds of bidders: an informed fund that *knows* it's worth \$22, and an uninformed account that just bids on everything. Both ask for \$100,000 of stock.

The informed fund only bids when the deal is good (worth more than the \$20 price), so it bids here, and because it showed honest early demand, the bank fills it generously — say 80% of its request, \$80,000, which immediately becomes worth $80{,}000 \times (22/20) = \$88{,}000$, a \$8,000 gain. The uninformed account bid the same \$100,000 but, because the deal is hot and oversubscribed, gets rationed to 20%, \$20,000, gaining $20{,}000 \times (22/20) - 20{,}000 = \$2{,}000$. On this *good* deal the uninformed still made money — just far less, because they were rationed exactly when it was worth being filled. The intuition: **even on the deals that work, the uninformed bidder is rationed down precisely where the fill was valuable, so the bank's allocation quietly transfers most of the pop to the informed accounts who revealed their signals.**

## Arena 2: Treasury auctions

Now to the cleanest, most explicit common-value auction a trader meets: the government bond auction. The US Treasury sells bills, notes, and bonds by auction roughly weekly, and the mechanics are a live demonstration of every idea in this post. For the plumbing of *why* and *when* the Treasury issues all this paper, see [Treasury issuance and the liquidity drain](/blog/trading/macro-trading/treasury-issuance-bills-coupons-liquidity-drain); here we focus on the *auction game* itself.

### Competitive vs non-competitive bids

There are two ways to bid in a US Treasury auction:

- **Non-competitive bid:** you say "I'll take up to \$X of this security at whatever yield the auction clears." You are guaranteed to be filled (up to \$10 million per auction for the public), and you pay the clearing price. You are explicitly *not* playing the bidding game — you've outsourced your price to the crowd. Small investors and TreasuryDirect users bid this way.
- **Competitive bid:** you specify the *yield* you're willing to accept. If your yield is at or below the clearing yield you're filled; if it's above, you get nothing. Primary dealers and large institutions bid competitively. *This* is where the winner's curse lives, because you can be filled at a price you set, and if you bid too aggressively (too low a yield, too high a price), you overpaid.

The non-competitive bid is, quietly, the *disciplined* move for someone with no edge: by accepting the clearing price you sidestep the curse entirely. You can't overpay relative to the field because you pay exactly what the field decided. The cost is you can't express a view; the benefit is you can't curse yourself.

### Uniform-price vs discriminatory: the format that tames the curse

The single most important design choice in a Treasury auction is the pricing rule, and it is a direct lever on the winner's curse.

In a **discriminatory** (pay-your-bid) auction — the format the US used before 1992 — each winning bidder pays *their own* submitted price. If you bid a price of 100.10 and the clearing price was 100.02, you pay 100.10 and you ate the extra 8 cents. Overbidding is punished one-for-one. The curse bites at full strength, so bidders shade hard and bunch their bids defensively near the expected clearing.

In a **uniform-price** (single-price) auction — which the US Treasury switched to for all marketable securities in 1998 after experiments starting in 1992 — *every* winner pays the same clearing price, the yield of the lowest accepted bid. Now a slightly-too-aggressive bid no longer costs you your own price; it costs you only the *market* price. The penalty for eagerness is muted, so bidders can afford to bid closer to their true estimate and reveal their real demand. The Treasury moved to uniform-price precisely because it *reduces the winner's curse*, which encourages more honest, aggressive bidding and — the Treasury's goal — lowers its borrowing cost.

![Two ways to settle a Treasury auction: in discriminatory pay-your-bid each winner pays their own price so overbidding is punished, while in uniform-price every winner pays one clearing price so the curse is softened](/imgs/blogs/the-winners-curse-in-ipos-treasury-auctions-and-mints-5.png)

The left column of the diagram (discriminatory) ends at "bidders shade hard and bunch bids near the expected clearing"; the right (uniform-price) ends at "bidders can bid nearer true value and reveal demand." That difference — softer curse, more honest bids — is the entire reason the format exists.

#### Worked example: shading in a Treasury auction

Suppose a 2-year note is truly worth \$1,000 per bond (a round-number stand-in for par plus accrued value), your estimate carries \$8 of noise, and you're competing against an effective field of $n = 8$ serious bidders. The model gives an expected overpayment of

$$8 \cdot \frac{8-1}{8+1} = 8 \cdot \frac{7}{9} = \$6.22.$$

In a **discriminatory** auction, where you pay your own bid, you should shade by the full \$6.22 and bid about \$993.78 — because any aggression you show, you pay for yourself. In a **uniform-price** auction, you can bid closer to fair value, because if you're slightly too high you still only pay the clearing price set by the marginal bidder, not your own number. The intuition: **the same \$6.22 curse demands hard shading when you pay your own bid and only gentle shading when everyone pays one price — which is exactly why the Treasury chose the gentler format.**

### Bid-to-cover: reading the demand signal

After every auction, the Treasury publishes a number that is catnip for traders: the **bid-to-cover ratio** — the total dollar amount bid divided by the amount actually sold. A bid-to-cover of 2.5 means \$2.50 of bids chased every \$1 of paper offered. It is a direct read on how crowded the field was, and therefore on how strong the curse was.

A *high* bid-to-cover (say above 3.0) means a hungry, crowded field — lots of aggressive bidders, a sharp curse, and a strong auction. A *low* bid-to-cover near 2.0 signals weak demand and raises the odds of a **tail** — an auction that clears at a yield noticeably *higher* (price lower) than where the security traded just before, meaning the Treasury had to concede a worse price to get the deal done. Recent US coupon auctions have typically printed bid-to-cover ratios in roughly the **2.4 to 2.6** range; a 2-year note auction near 2.6 is healthy, while a 30-year bond limping in below 2.3 with a long tail makes headlines and can nudge the whole curve.

![Bid-to-cover is total bids divided by the amount offered; a ratio near two signals soft demand and a likely tail while a ratio above three signals a crowded hungry field where the curse is sharpest](/imgs/blogs/the-winners-curse-in-ipos-treasury-auctions-and-mints-7.png)

The matrix reads the three regimes. A ratio near 2.0 (top row, red) is thin demand — soft, tail-prone, and you can bid nearer fair value because the field is light. A ratio near 2.5 (middle, the recent norm) is healthy and normal. A ratio above 3.0 (bottom, the crowded case) is where the curse is sharpest: an eager crowd is overpaying, and a disciplined bidder shades harder. Note the symmetry with our model: high bid-to-cover *is* a high effective $n$, and high $n$ means a bigger required shave.

### Primary dealers: the obligated bidders

One more institutional wrinkle. The Treasury anoints a set of large banks as **primary dealers** (about two dozen at any time), and a primary dealer is *obligated* to bid in every auction — to put in a meaningful, reasonable bid for the security being sold, come what may. They cannot simply sit out a deal they dislike. This is a clever piece of mechanism design: it guarantees the Treasury a floor of demand, so even a cold auction clears. But it also means primary dealers are structurally exposed to the curse — they must bid even when they'd rather not, so they must be *especially* disciplined shaders, and they hedge their auction exposure aggressively in the futures and repo markets. The obligation is the price of the franchise; the shading is how they survive it.

### The when-issued market: shrinking the noise before the auction

There's an elegant institutional fix for the curse that's worth understanding, because it's a real-world example of *reducing the noise* $w$ rather than just shading harder. In the days before a Treasury auction, the security trades on a **when-issued** (WI) basis — a forward market in the not-yet-issued bond, settled when the auction completes. The WI market lets every dealer see a continuously-updated consensus price *before* they have to submit their sealed bid.

Why does this matter? Recall the model: the curse is $w \cdot \frac{n-1}{n+1}$, and $w$ is the noise in *your* estimate. The WI market lets you anchor your estimate to a public, liquid consensus, which slashes the dispersion of estimates across bidders. When everyone's signal is anchored to the same WI price, the spread of bids collapses, the "highest bidder" isn't much above the rest, and the curse shrinks toward zero. The WI market is, in effect, a noise-reduction machine bolted onto the auction. It's why Treasury auctions — despite being huge common-value auctions — usually clear within a basis point or two of the pre-auction price: the format and the WI market together have squeezed most of the curse out. Compare that to an NFT mint, which has no WI market, no consensus price, and therefore maximal noise and a maximal curse. The presence or absence of a pre-auction price-discovery market is one of the biggest determinants of how badly the curse bites.

#### Worked example: what the when-issued market does to your shave

Suppose without any pre-auction information your estimate of a note's value carries \$8 of noise, and you face $n = 8$ bidders. From earlier, your shave is $8 \cdot \frac{7}{9} = \$6.22$. Now the WI market trades all morning and tightens the consensus, so your effective noise drops to \$2. Recompute: $2 \cdot \frac{7}{9} = \$1.56$. Your required shave fell from \$6.22 to \$1.56 — a 75% reduction — *purely because the noise shrank, with the field size unchanged*. The intuition: **price-discovery before the auction attacks the curse at its root by shrinking the disagreement among bidders, which is why liquid, well-followed instruments with a pre-auction market suffer far less curse than opaque, hyped ones with none.**

## Arena 3: NFT mints and token sales

Crypto reinvented the auction from scratch, usually without realizing it was re-deriving century-old auction theory — and it re-derived the winner's curse the hard way. The lessons here are about *defense*: knowing the game so you are not the cursed winner. (For the chain-level mechanics of how transactions get ordered and front-run, see the on-chain MEV material; here we stay on the bidding game.)

### Gas auctions: an accidental first-price auction

When an NFT collection "mints" — opens for sale at a fixed price, first-come-first-served — and demand wildly exceeds supply, the *real* contest is not the mint price; it's the **gas auction** to get your transaction included in the next block. On Ethereum, you bid a *priority fee* (a tip to the validator) for your transaction, and the highest tips get in first. That is a sealed-bid, pay-your-bid (discriminatory) auction for block space — the harshest format for the curse.

During the worst mint frenzies, gas fees spiked to hundreds of dollars per transaction. The people who "won" — who got their transaction included and minted the NFT — were the ones who bid the highest gas, i.e., the most over-eager estimators of how valuable getting in early would be. Many paid \$300+ in gas to mint an NFT that traded below mint price an hour later, *and lost the gas even on failed transactions*. The gas auction is the winner's curse with extra cruelty: you can pay the bid and still not win the item.

### Dutch-auction mints: the format that was supposed to help

To tame the gas wars, many projects moved to **Dutch-auction** mints. Recall the format: the price starts high and *falls* over time until buyers step in; the first to accept pays the current (high) price, and as more sell out the price keeps dropping. The theory is elegant — it sets a fair clearing price without a gas war, because there's no rush to be *first* at a low price.

In practice, Dutch auctions still curse the eager. Whoever buys *first* pays the *highest* price on the descending curve — and the bidders who buy first are, by selection, the most optimistic about the project. Art Blocks and several large drops used Dutch auctions; the early buyers routinely paid multiples of where the price settled. The format changed *who* pays the curse (the impatient, optimistic early buyers rather than the highest gas bidder) but did not abolish it. The disciplined move in a Dutch auction is to *wait* — let the price fall toward the level where calm, informed buyers step in — which is exactly bid shading expressed as patience.

### Why hype maximizes the curse

Return to the model: the curse is $w \cdot \frac{n-1}{n+1}$, growing in both noise $w$ and field $n$. A hyped mint maximizes *both* at once. Hype draws an enormous crowd (huge $n$) and surrounds an asset whose true value is almost completely unknown (huge $w$). The product is a monstrous curse. This is the structural reason hyped mints so reliably curse the winners: hype is, in the language of this post, a machine for simultaneously inflating the field size and the estimate noise.

### Token sales and the ICO curse

NFT mints are the vivid case, but the same structure governs **token sales** — ICOs (initial coin offerings), IDOs (decentralized exchange offerings), and the various sale formats crypto has churned through. A token has one eventual market value, and everyone buying in the sale is estimating it under enormous noise, because a pre-launch token has essentially no fundamentals to anchor on — just a whitepaper, a team, and a community's collective hope. That is the maximum-noise regime, $w$ as large as it ever gets.

Several token-sale formats are *literally* auctions, and they curse the eager in textbook ways. The **batch auction** or **liquidity bootstrapping pool** format (used by some launches) starts the token price high and lets it drift down as buyers come in — a Dutch auction by another name, which curses the impatient first-buyers exactly as the NFT Dutch auctions do. The **fixed-price-plus-allocation-lottery** format (a fixed sale price, with a lottery deciding who gets in when oversubscribed) reproduces the IPO allocation problem precisely: you win the lottery and get filled most easily on the sales nobody fought over, which are the ones the informed money skipped. The **first-come-first-served** sales reproduce the gas war.

The defensive read across all of them is the same: a token sale you can easily get into, in full size, with no competition, is a token sale the informed money looked at and walked away from. As with IPO allocation, *ease of filling is a bad signal*. The sales that are impossible to get into — that sell out in seconds, that you get rationed on — are the ones the smart money fought for. This is the winner's curse and the lemons problem braided together, and the only winning move is often to recognize the structure and not play: the disciplined "bid" in a maximum-noise, maximum-field sale is frequently zero.

#### Worked example: the cursed mint

You're eyeing a hyped NFT mint. Your honest estimate of the eventual floor price is 1.0 ETH, but the noise is enormous — call it $w = 0.6$ ETH, because nobody really knows. The field is huge: effectively $n = 50$ aggressive minters chasing each slot. The model:

$$\text{overpayment} = 0.6 \cdot \frac{50-1}{50+1} = 0.6 \cdot \frac{49}{51} = 0.576 \text{ ETH}.$$

The naive winner pays about $1.0 + 0.576 = 1.58$ ETH (in mint price plus gas) for something worth 1.0 ETH — an overpayment of more than half the item's value. To break even you'd have to bid as if your estimate were 0.42 ETH, not 1.0. The intuition: **a hyped mint is the worst-case auction — maximum field, maximum noise — so the only disciplined bids are a brutal shave or, more honestly, no bid at all.** This is the detection-and-defense lesson: recognizing the cursed structure is what keeps you from being the eager winner who funds everyone else's exit.

## Common misconceptions

**"More competition means I should bid higher to win."** This is the single most expensive instinct in auctions, and it is backwards for common-value items. More bidders means the winning estimate is more extreme, so the curse is worse, so you must shade *more*. Our table showed the required shave nearly tripling from \$6.67 (2 bidders) to \$18.10 (20 bidders). The correct response to a crowded room is restraint, not aggression. Bidding higher because there's more competition is volunteering to be the cursed winner.

**"I got my full IPO allocation — great, I must have gotten lucky."** A full allocation is the market telling you the informed money didn't want this deal. On hot deals you're rationed to a sliver; on cold deals you're filled in full. A consistently *full* allocation across deals means your book is selecting for the rejects. The worked example showed a naive +5% average turning into a −15.5% allocation-weighted reality.

**"The winner's curse only matters for oil leases and academic auctions."** It governs every common-value auction a trader touches: IPOs, Treasury bills, corporate bond new-issues, NFT mints, token sales, even the order book (a fast fill is a one-sided auction you won by being the most eager). The arenas differ; the selection effect is identical. If there's one true value and noisy estimates, the curse is present.

**"Uniform-price auctions remove the winner's curse."** They *soften* it, they don't remove it. In a uniform-price auction you still face the selection effect — winning still means you were optimistic — but the *penalty* is muted because you pay the clearing price, not your own bid. The curse is in the *information* (winning is bad news); the format only changes how much that bad news costs you. You still shade; you just shade less.

**"If the IPO pops 18% on average, IPOs are a free lunch."** The pop is real but conditional on getting filled, and you get filled least on the biggest poppers. The 18% is a statistic about shares *offered*; your realized return is allocation-weighted and dragged toward the cold deals. Underpricing is precisely the compensation calibrated to keep you *just barely* willing to play after the curse — not a windfall.

**"In a Dutch auction I should buy early to lock in the NFT."** Buying early in a descending-price auction means paying the *highest* price on the curve, and the early buyers are selected for optimism. The disciplined move is patience — let the price fall toward where calm buyers step in. Eagerness in a Dutch auction is bid-shading in reverse: you're *adding* the curse instead of subtracting it.

**"The bid-to-cover ratio tells me whether the bond is a good buy."** It tells you how *crowded* the auction was, not whether the price was right. A high bid-to-cover means a hungry field — which, by our model, means a *bigger* curse, not a better deal. Strong demand at the auction is information about competition, and competition is the thing you must shade *against*. Reading a high bid-to-cover as "great, everyone wants it, I should want it too" is precisely the more-competition-so-bid-higher mistake in macro clothing.

**"Bid shading just means I'll lose every auction to more aggressive bidders."** You will lose *more* auctions — that's the point. You win selectively, on the auctions where even the shaded bid clears, which are the ones where the field was thin or your edge was real. The bidder who wins *every* auction is the one paying the curse on all of them. In auctions, a lower hit rate at better prices beats a high hit rate at cursed prices, the same way a market maker would rather skip a trade than fill an adverse-selected one.

## How it shows up in real markets

**Facebook IPO (May 2012).** Priced at \$38 amid frenzied retail demand; many retail bidders got full allocations precisely because institutions were wary at that price. The stock fell to roughly \$18 within three months — more than 50% — before eventually recovering over years. The retail bidders who got their full fill were the adverse-selected losers; the muted institutional demand was the informed signal they ignored. A full fill on a hyped deal is a warning, and Facebook is its monument.

**The winner's curse coined in oil (1971).** The term comes from three Atlantic Richfield engineers — Capen, Clapp, and Campbell — who noticed that oil companies winning offshore lease auctions in the Gulf of Mexico were systematically earning poor returns. The winners were the firms whose geologists most over-estimated the reserves. Their paper formalized the cure: shade your bid below your engineering estimate, and shade more as the number of competing bidders rises. Half a century later the same math prices a Treasury bill.

**US Treasury's switch to uniform-price (1992–1998).** After the 1991 Salomon Brothers auction-rigging scandal, the Treasury began experimenting with uniform-price (single-price) auctions for 2- and 5-year notes in 1992, and adopted the format for *all* marketable securities by 1998. The documented rationale was reducing the winner's curse to encourage broader, more aggressive, more honest bidding — directly lowering the government's borrowing cost. It is the largest real-world A/B test of auction theory ever run, and it confirmed that softening the curse brings in better bids.

**Weak Treasury auctions move the whole market.** A soft auction with a low bid-to-cover and a long *tail* (clearing well below pre-auction levels) signals that demand for government debt is thin — and it ripples instantly. Several 2023 long-bond auctions printed disappointing bid-to-cover ratios and tails, knocking bond prices lower and yields higher across the curve in the minutes after the 1pm results. The bid-to-cover is the field-size signal of our model made public, and a thin field with a soft clear is the market pricing weak demand in real time.

**The 2021 NFT gas wars.** During peak mania, marquee mints (Bored Ape adjacents, various PFP drops) triggered gas auctions that pushed Ethereum priority fees into the hundreds of dollars per transaction. Minters paid \$200–\$500 in gas — and lost it even on *failed* transactions that didn't make the block — to mint NFTs that frequently traded below cost within hours. It was a textbook discriminatory common-value auction with maximal noise and a maximal field: the perfect storm for the curse, paid in burnt gas.

**Dutch-auction NFT drops (Art Blocks and beyond).** Projects adopted Dutch auctions to kill the gas wars, and it worked for *that* problem. But the early buyers — selected for optimism and impatience — routinely paid multiples of the settled clearing price as the curve descended. The format moved the curse from "highest gas bidder" to "most eager early buyer," confirming that auction format changes *who* gets cursed, not *whether* someone does. The disciplined minters who waited for the price to fall paid far less.

**Google's Dutch-auction IPO (2004).** Google deliberately ran its IPO as a modified Dutch auction to *reduce* underpricing and the bank-controlled allocation game — to let the auction find a fair price and hand the "pop" to the company rather than to favored institutions. It priced at \$85 and rose more modestly on day one than a typical bookbuilt hot deal. It's the clearest case of a company using auction *design* to fight the structural curse, choosing a format that reveals demand over one that rations allocations to insiders.

## The playbook: how to bid without cursing yourself

This is the part that turns the model into a habit. Whenever you find yourself in a common-value auction — and you are in one far more often than you think — run the disciplined bidder's checklist.

**1. Identify that it *is* a common-value auction.** Ask: is there one true value everyone's estimating, or do we each value it differently? IPO, Treasury, bond new-issue, NFT mint, token sale, a fast fill on a resting order — all common-value. If yes, the curse is present and you must shade.

**2. Estimate the value, then *subtract* the curse.** Your bid is *not* your estimate. Your bid is your estimate minus the expected overpayment of the winner. Two inputs set that shave: the field size $n$ and your noise $w$. Bigger field, bigger shave; noisier asset, bigger shave. The shave is $w \cdot \frac{n-1}{n+1}$.

![The disciplined bid is your estimate minus the curse shave; against a larger field the held-back shave grows and the bid you actually submit falls from about ninety-three to eighty-two dollars](/imgs/blogs/the-winners-curse-in-ipos-treasury-auctions-and-mints-6.png)

The bar chart is the playbook in one image. Each bar is your \$100 estimate, split into the blue *bid you actually submit* and the amber *shave you hold back* for the curse. Against 2 bidders you bid \$93.33 and hold back \$6.67; against 20 you bid \$81.90 and hold back \$18.10. The dashed line is your unshaded estimate — the bid the naive, cursed player would submit. Disciplined bidding is the gap between the dashed line and the top of the blue.

#### Worked example: running the full checklist on a bond new-issue

You're allocated room to bid on a new corporate bond. Your desk's model says fair value is \$100. The issue is somewhat illiquid and your estimate noise is \$15. You judge the effective competitive field at $n = 5$ serious accounts. Run it:

$$\text{shave} = 15 \cdot \frac{5-1}{5+1} = 15 \cdot \frac{4}{6} = \$10.00.$$

So you bid \$90, not \$100. If the field turns out larger — word gets around and it's really $n = 20$ accounts — recompute: $15 \cdot \frac{19}{21} = \$13.57$, so bid \$86.43. If you instead *misjudge* and bid your \$100 estimate, you win exactly the deals where you were most over-optimistic and bleed the \$10–\$14 of curse. The intuition: **the disciplined bid is a single subtraction — estimate minus shave — but skipping it is how desks quietly lose money on every new issue they "win."**

**3. Use the format and the obligation to your advantage.** If you have no edge, take the non-competitive route (the Treasury non-competitive bid, the index-fund approach, simply not chasing the mint) and accept the clearing price — you can't curse yourself if you pay what the field decided. If you must bid (you're a primary dealer, you run the book), shade *hardest* in pay-your-bid formats and somewhat less in uniform-price.

**4. Read the field-size signal.** Bid-to-cover, IPO oversubscription, NFT mempool depth, the gas tip distribution — these are all live readouts of $n$. A crowded field is a *reason to bid less*, not more. When everyone is eager, the curse is at its worst, and the disciplined response is to stand further back.

**5. Know who's on the other side and what they know.** This is the spine of the whole series. In an auction, the people most eager to outbid you are, by selection, either better-informed (you're adverse-selected) or more over-optimistic (they're about to be cursed — don't join them). Your edge is not a better point estimate; it's the *discipline to shade* when the room won't. The bidders who think in terms of the winner's curse — and who know the others won't — are the ones who get filled at prices that actually make money.

**Invalidation.** This model assumes a genuine common value and roughly symmetric, unbiased signals. If you have a *real informational edge* — you actually know the value better than the field — the curse shrinks toward zero and you can bid more aggressively; the shave is for your *ignorance*, and an edge is the opposite of ignorance. And if the auction is genuinely private-value (you uniquely value the thing), shade little or not at all. The discipline is to be honest about which world you're in, because the most dangerous bidder is the one who *thinks* he has an edge and is really just the most optimistic estimator in a common-value room.

One last sizing note, because the curse interacts with how much you bid *for*, not just the price you bid *at*. In auctions where you can win multiple units — Treasury auctions, large IPO allocations, token sales — winning a *bigger* quantity is itself information: you got the most when others wanted the least. So the curse should scale your *size* as well as your price. A disciplined bidder asks for less size in auctions that look crowded and hyped (high $n$, high $w$), and is willing to take real size only when the field is thin or the noise is genuinely low and they have done the work to shrink their own $w$ below the field's. The two knobs — how much to bid, and how much to bid *for* — both turn the same way: away from the eager crowd. That is the entire discipline, and it is the difference between being the house and being the cursed winner who funds it. Estimate honestly, subtract the curse, shade harder as the room fills, and let the over-optimistic bidders win the auctions you were right to lose.

## Further reading & cross-links

- [Adverse selection and the winner's curse: why a fast fill is bad news](/blog/trading/game-theory/adverse-selection-and-the-winners-curse-why-a-fast-fill-is-bad-news) — the same curse applied to order flow, where this series first introduced it. A fast fill is a one-bidder auction you won by being the most eager.
- [Asymmetric information: the lemons problem in markets](/blog/trading/game-theory/asymmetric-information-the-lemons-problem-in-markets) — the deeper root: when one side knows more than the other, the uninformed get selected into the worst deals. IPO allocation is a lemons market.
- [The bid-ask spread as an adverse-selection game (Glosten-Milgrom)](/blog/trading/game-theory/the-bid-ask-spread-as-an-adverse-selection-game-glosten-milgrom) — how a market maker prices the same selection risk continuously into the spread instead of in a discrete auction.
- [Treasury issuance: bills, coupons, and the liquidity drain](/blog/trading/macro-trading/treasury-issuance-bills-coupons-liquidity-drain) — the macro plumbing of *why* the Treasury auctions so much paper and what all that issuance does to market liquidity.
- [Who is on the other side of your trade?](/blog/trading/game-theory/who-is-on-the-other-side-of-your-trade) — the foundational question this whole series answers, and the first thing to ask before you submit any bid.

*This is educational material about how auctions and the winner's curse work, not individualized financial advice. Every auction that can make you money can lose it; the curse is exactly the mechanism by which winning bids lose.*
