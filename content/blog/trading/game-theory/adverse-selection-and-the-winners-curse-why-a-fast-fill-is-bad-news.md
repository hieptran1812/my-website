---
title: "Adverse Selection and the Winner's Curse: Why a Fast Fill Is Bad News"
date: "2026-06-18"
publishDate: "2026-06-18"
description: "When the act of trading is itself information, an instant fill warns you that you were picked off, and winning an auction means you probably overpaid unless you shave your bid."
tags: ["game-theory", "trading", "adverse-selection", "winners-curse", "market-microstructure", "order-flow", "market-making", "auctions", "information-asymmetry"]
category: "trading"
subcategory: "Game Theory"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — A trade is a strategic interaction, so the fact that someone agreed to take the other side of yours is information; once you read that information, a fast fill and a won auction stop looking like wins.
>
> - **Adverse selection**: when one side knows more, the eagerness of your counterparty is a warning. The people most willing to sell you something cheap are disproportionately the ones who know it is worth less.
> - **The fast-fill tell**: a resting limit order that fills instantly and keeps trading *through* your price was probably picked off by someone informed (toxic flow). A fill that sits, then comes back to you, is more likely benign.
> - **The winner's curse**: in a common-value auction, winning means you were the most optimistic estimator — so you systematically overpaid. With 10 noisy bidders and a true value of \$100, the naive winner overpays by about \$16. The defense is to *shave* your bid below your own estimate.
> - **The one rule**: bid your signal minus the shave, and treat speed-and-aggression on the other side of your quote as a reason to widen, not to celebrate.

In late September 2010, a trader at a Chicago proprietary firm described a fill that still bothered him a decade later. He had a resting buy order — a *limit order*, an instruction to buy only at his price or better — sitting on an exchange at \$41.00 for a mid-cap stock. The market had been drifting around \$41.05 all morning, quiet. Then, in the space of about forty milliseconds, his entire order filled at \$41.00, and the next prints he saw were \$40.92, then \$40.85, then \$40.70. He had bought exactly at the moment the floor fell out. The fill felt like luck — he got his price — right up until the price kept going and he realized he had been *handed* the stock by someone who knew it was about to drop.

That is the whole post in one anecdote. When a stranger is unusually eager to trade with you, the eagerness itself carries information, and almost always it carries *bad* information for you. A trade is not a bet against an indifferent nature that flips a coin; it is a deal with a specific human or algorithm who chose to take your side for a reason. The question that should run through your head on every fill is the one this whole series keeps coming back to: *who is on the other side, and what do they know that I don't?*

The diagram below is the mental model for the entire post: the same resting order, the same \$100 mid-price, two different fills. On the left, the toxic version — instant fill, market trades through, you are upside down within seconds. On the right, the benign version — a fill that sits, then mean-reverts back. The job of this post is to explain *why* the left column is the one you should expect when you are the least-informed person in the room, and what to do about it.

![A fast fill that keeps trading through your level is toxic, while a slow fill that mean-reverts is benign](/imgs/blogs/adverse-selection-and-the-winners-curse-why-a-fast-fill-is-bad-news-1.png)

We will build two linked ideas from nothing. The first is **adverse selection**: the systematic tilt that appears whenever one side of a trade knows more than the other, so that the *willingness to trade* is itself a signal. The second is the **winner's curse**: the specific, quantifiable penalty you pay for *winning* a contest to buy something whose value nobody knows for sure. They are two faces of the same coin — the coin being that in markets, getting what you asked for is often the bad outcome.

## Foundations: information asymmetry, the lemons problem, and common-value auctions

Before anything else we need three plain-English definitions. Everything in the post hangs off them, so we go slowly.

### Information asymmetry: one side knows more

**Information asymmetry** is the simple fact that the two sides of a trade rarely know the same things. The person selling you a used car knows whether it stalls in the rain; you don't. The trader selling you 10,000 shares may have seen an order-flow pattern, a news headline, or a model output that you haven't. A *symmetric* market is one where both sides have the same information and trade only because they disagree about the future or have different needs (one wants cash now, the other wants to invest). An *asymmetric* market is one where one side trades *because* it knows something the other doesn't.

The word that matters is "because." If the only reason someone is selling to you is that they need the money, that's fine — you can both be happy. If part of the reason they are selling is that they know the thing is overpriced, then their sale is partly a transfer of their bad news onto you, dressed up as a normal trade.

### The market for lemons: how asymmetry can unravel a whole market

The cleanest illustration is the 1970 paper that won George Akerlof a share of the 2001 Nobel Prize in economics, "The Market for 'Lemons'." A "lemon," in American slang, is a defective used car. Akerlof's question was deceptively small: why is a brand-new car worth so much less the moment you drive it off the lot, even with zero miles?

Here is the logic, built one step at a time. Suppose half the used cars of a given model are good (worth \$10,000 to a buyer) and half are lemons (worth \$4,000). The sellers know which is which; the buyers can't tell them apart. A buyer who can't distinguish them will only pay for the *average* — somewhere around \$7,000. But now look at who is willing to sell at \$7,000. An owner of a good car worth \$10,000 will not sell it for \$7,000; they'd rather keep it. An owner of a lemon worth \$4,000 is *delighted* to get \$7,000. So at a \$7,000 price, the good cars withdraw and only lemons are offered. Buyers, who are not stupid, notice that the cars actually for sale are mostly lemons, so they lower their offer toward \$4,000. That drives out any remaining good-ish cars, and the discount deepens. The market can collapse entirely — good cars become almost unsellable not because they're bad but because buyers can't tell, and the act of offering to sell is itself a slightly bad signal.

That self-reinforcing collapse — discount, good sellers leave, average drops, discount deepens — is the picture below. It is the prototype of adverse selection: the *selection* of who chooses to trade is *adverse* to the uninformed side.

![A pool of mixed-quality sellers where good sellers exit until only lemons remain](/imgs/blogs/adverse-selection-and-the-winners-curse-why-a-fast-fill-is-bad-news-7.png)

The phrase to carry forward: **adverse selection is when the willingness to trade is correlated with bad news for you.** The eager seller is, on average, the informed seller. Keep that sentence; we are going to apply it to order flow in a moment.

#### Worked example: how the lemons pool poisons the price

Put real numbers on the unraveling so you can see the discount feed on itself. Start with 100 cars: 50 good (worth \$10,000) and 50 lemons (worth \$4,000). A buyer who can't tell them apart values the *average* car at $0.5 \times 10{,}000 + 0.5 \times 4{,}000 = \$7{,}000$, so they offer \$7,000.

At \$7,000, every good-car owner refuses (their car is worth \$10,000 to them) and every lemon owner sells (theirs is worth \$4,000). So the 50 cars that actually sell are *all lemons*. The next round of buyers, having learned that the pool on offer is now all lemons, will only pay \$4,000. The \$7,000 price was never an equilibrium — it was a way-station on the slide from \$7,000 down to \$4,000. The good cars, worth \$10,000 and genuinely better, simply leave the market: there is no price at which their owners will sell and informed-discounting buyers will buy. A market with \$300,000 of good cars and \$200,000 of lemons collapses to a market that trades only \$200,000 of lemons.

The intuition: when buyers must price for the average but sellers self-select on quality, the average itself rots downward until only the worst remains.

### Signaling: how a market fights back against the lemons problem

If the story stopped there, no market with asymmetric information could ever function — and yet used cars *do* sell, blocks of stock *do* trade. The reason is **signaling**: the informed side finding a credible, costly way to prove quality, so the good types can separate themselves from the bad. A signal only works if it's *cheaper for the good type than the bad type* — otherwise the lemons would just copy it.

In the used-car market, the signal is the **warranty**: a seller who offers a money-back guarantee is, in effect, betting that the car won't break, which a lemon owner can't afford to do. The warranty is cheap for a good-car owner (the car won't break, so they never pay out) and expensive for a lemon owner (it will, so they will). That asymmetry makes the warranty *credible*, and it lets good cars trade at good prices. Certified pre-owned programs, inspection reports, and dealer reputation are all the same move: a costly signal that separates the good from the bad.

The market analog is everywhere. A company that trades patiently and transparently signals "I'm a liquidity need, not informed." A firm that submits to an independent audit signals "my books are real." An issuer that retains a slice of its own securitization (skin in the game) signals "I believe this pool is good." Every one of these is a costly signal designed to defeat the lemons unraveling — to let the uninformed side trust that the eager counterparty isn't eager *because* they know something bad. When you trade, the absence of any such signal — an anonymous, urgent counterparty with no skin in the game — is itself a small red flag.

### Common-value vs private-value auctions

The second idea needs a different setup: an **auction**, a contest where multiple people bid for one item and the highest bid wins. Auctions come in two flavors, and the distinction is everything.

A **private-value** auction is one where the item is worth a different, personal amount to each bidder, and your value doesn't depend on anyone else's. A signed concert poster, a painting you love, a house you want to live in — your value is *yours*. If you win, you got the thing you valued, and there's no hidden problem.

A **common-value** auction is one where the item has a single true value that is the *same* for everyone, but nobody knows it exactly — each bidder only has a noisy estimate. The classic example is bidding for the oil under a tract of land. The oil is worth whatever it's worth — a fixed dollar amount — but each oil company only has its own geologists' estimate, which is high or low by some random error. A Treasury bill at auction, a block of shares whose fair value is "whatever the market clears at next week," a startup's equity, a takeover target — these are common-value. Everyone is trying to estimate *the same number*, and they have different noisy guesses.

The winner's curse lives entirely in the common-value world. When the value is common but the estimates are noisy, *winning is bad news about your estimate*, and that is the engine of everything that follows.

### The solution concept: think one level deeper

This series treats trading as a game, and the recurring discipline is to reason one level deeper than your counterparty. In a private-value auction, the naive strategy (bid your value) is roughly fine. In a common-value auction, the naive strategy is a trap, and the only way to see the trap is to ask the game-theoretic question: *given that I won, what does that tell me about my own bid?* That conditional — "given that I won" — is the whole trick. We define it precisely next.

## The winner's curse, derived from zero

Let's build the winner's curse with numbers small enough to hold in your head, then read the result off the model.

### The setup

There is one item with a true common value $v$. We'll use $v = \$100$ throughout — say, the fair value of a block of shares. There are $n$ bidders. Each bidder $i$ sees a private signal

$$s_i = v + e_i$$

where $e_i$ is that bidder's estimation error — sometimes too high, sometimes too low, averaging zero. We'll model each error as uniform on the interval from $-w$ to $+w$, so $w$ is the *noise*: the half-width of how wrong your estimate can be. If $w = \$20$, your estimate of the \$100 value could be anywhere from \$80 to \$120, all equally likely.

The naive bidder does the obvious thing: bids their signal. If your geologists say the oil is worth \$108, you bid \$108. Reasonable, right? Your estimate is unbiased — on average it equals the truth.

### Why the average is the wrong thing to think about

Here is the subtlety that traps almost everyone the first time. Your estimate is unbiased *unconditionally* — averaged over all the situations you could be in. But you don't win every auction. You only win when your bid is the *highest*. And the bid is highest precisely when your estimate $s_i$ happened to be the most optimistic of the bunch. So the auctions you win are a biased sample: they are the ones where you, specifically, overestimated.

Said as the sentence to remember: **you are not the average bidder when you win; you are the most optimistic bidder.** Winning *selects* for having overestimated. The expected value of your signal *given that it was the maximum of $n$ draws* is well above the true value. That gap is the curse.

For uniform errors, there's a clean formula for the expected highest error among $n$ draws:

$$E[\max e] = w \cdot \frac{n-1}{n+1}$$

The naive winner's expected bid is therefore $v + w \cdot \frac{n-1}{n+1}$, and the **expected overpayment** — how much the naive winner pays above true value — is exactly that $w \cdot \frac{n-1}{n+1}$ term. This is precisely what `data_gametheory.winners_curse(true_value, n_bidders, signal_noise)` computes; every overpayment number below comes from that function, not from anything I made up.

#### Worked example: two bidders, \$20 of noise

Start with the smallest interesting case. True value $v = \$100$, noise $w = \$20$, and $n = 2$ bidders.

The expected maximum error is $w \cdot \frac{n-1}{n+1} = 20 \cdot \frac{1}{3} = \$6.67$. So the winner — the one who drew the higher estimate — bid about \$106.67 on average for something worth \$100. They overpaid by **\$6.67**. Even with just two bidders and a fairly modest estimation error, winning costs you almost 7% of the item's value, purely because winning means you were the higher of the two guesses.

The intuition: even a two-horse race is a contest your optimism wins, and your optimism is exactly what makes you overpay.

#### Worked example: ten bidders, same noise

Now crowd the auction. Same $v = \$100$ and $w = \$20$, but $n = 10$ bidders.

$$E[\max e] = 20 \cdot \frac{10-1}{10+1} = 20 \cdot \frac{9}{11} = \$16.36$$

The naive winner now bids about \$116.36 for the \$100 block — an overpayment of **\$16.36**, more than 16% of value. Adding bidders made the curse much worse, because the maximum of ten noisy guesses is far more extreme than the maximum of two. The chart below plots this exact relationship from the model: expected overpayment climbing as the number of bidders grows, approaching but never quite reaching the noise width $w = \$20$ as a ceiling (the most optimistic of infinitely many estimators would have erred by the full half-width).

![Expected overpayment of the naive winner rising with the number of bidders toward the noise ceiling](/imgs/blogs/adverse-selection-and-the-winners-curse-why-a-fast-fill-is-bad-news-2.png)

Read the shape carefully, because it is counterintuitive. In a *private*-value auction, more bidders is good for the seller and irrelevant to your strategy. In a *common*-value auction, more bidders is *dangerous to you as a buyer*: every extra rival raises the bar you have to clear to win, and clearing a higher bar means you must have been more optimistic, which means you overpaid more. The crowd is not your friend. When you find yourself in a bidding war over something whose true value is uncertain and shared, the right reaction is not excitement but suspicion.

Notice also the *shape* of the curve: it rises steeply at first and then flattens. Going from 2 to 3 bidders adds a lot of curse (\$6.67 to \$10.00 in our example); going from 50 to 100 adds almost nothing (\$19.22 to \$19.60). The reason is that the maximum of many draws is bounded by the noise width — the most optimistic of even a thousand bidders can only have erred by up to the full half-width $w$. So the curse saturates: the first few rivals do most of the damage, because each one is fairly likely to out-optimist you, while the hundredth rival rarely sets a new high. The practical reading is that you don't need a *huge* crowd for the curse to bite hard — even three or four serious rivals already put most of the overpayment on the table, so the moment a contest becomes genuinely competitive, the shave matters.

### How noise makes it worse

The other lever is the noise $w$ itself — how uncertain your estimate is. The overpayment is *proportional* to $w$: double your estimation error and you double the curse. This is the most actionable fact in the whole model, because noise is the one input you partly control. A bidder with a tighter estimate (better geologists, better models, faster data) suffers less from the curse and can bid more aggressively without getting burned.

#### Worked example: the same curse from two different sources

Compare two situations that produce the *identical* overpayment, to feel how the two levers trade off. Situation A: $n = 5$ bidders, noise $w = \$10$. Situation B: $n = 2$ bidders, noise $w = \$20$.

- Situation A: $E[\max e] = 10 \cdot \frac{5-1}{5+1} = 10 \cdot \frac{4}{6} = \$6.67$.
- Situation B: $E[\max e] = 20 \cdot \frac{2-1}{2+1} = 20 \cdot \frac{1}{3} = \$6.67$.

Both overpay by exactly **\$6.67**. A crowded auction with sharp estimates can be just as cursed as a quiet auction with sloppy ones. The lesson is that you must account for *both* — how many rivals you face *and* how noisy your own read is — and the chart below shows the noise lever directly: for any fixed number of bidders, overpayment rises in a straight line as your estimation error grows.

![Expected overpayment rising linearly with signal noise, steeper for more bidders](/imgs/blogs/adverse-selection-and-the-winners-curse-why-a-fast-fill-is-bad-news-3.png)

The takeaway from the picture: if you can't sharpen your estimate (reduce $w$), you must protect yourself another way — by bidding less than you think it's worth. That defense is the shave.

### The shave: bidding below your own estimate

The fix for the winner's curse is to bid your signal *minus a correction*. You assume that *if* you win, your signal was probably on the high side, so you pre-emptively subtract the expected overestimate. The amount you subtract — the **shave** (also called *bid shading*) — is exactly the expected overpayment we just computed: $w \cdot \frac{n-1}{n+1}$.

The logic is almost philosophical: you bid as if you have already won and are accounting for the bad news that winning implies. You discount your own enthusiasm in advance. A disciplined bidder doesn't ask "what do I think it's worth?" They ask "what should I bid so that, *conditional on winning*, I break even on average?"

#### Worked example: the bid you should actually submit

Suppose your own signal lands right at \$100 and you face $n = 10$ rivals with noise $w = \$20$. Naively you'd bid \$100. But the shave is

$$w \cdot \frac{n-1}{n+1} = 20 \cdot \frac{9}{11} = \$16.36,$$

so the disciplined bid is $100 - 16.36 = \$83.64$. You submit \$83.64 for something your own best guess says is worth \$100. It *feels* like leaving money on the table — and most of the time you'll lose the auction to someone who shaded less and will regret it. But across many auctions, the shave is what keeps you from systematically overpaying. The bar chart makes the rule concrete across crowd sizes: the naive bid is a flat \$100, while the disciplined bid drops further below it as the field gets larger.

![Naive bid fixed at one hundred dollars versus a disciplined bid that shaves deeper as bidders increase](/imgs/blogs/adverse-selection-and-the-winners-curse-why-a-fast-fill-is-bad-news-6.png)

The bid you submit at $n = 2$ is \$93.33 (a \$6.67 shave); at $n = 5$ it's \$86.67; at $n = 10$ it's \$83.64; at $n = 20$ it's \$81.90. More rivals, deeper shave. The one-sentence intuition: in a common-value contest, the right amount to *want* to win is much less than the right amount to *think it's worth*.

## From auctions to order flow: every fill is a tiny auction you might have won

Now the bridge. The winner's curse and adverse selection are usually taught as separate topics — one about auctions, one about used cars. In markets they are the same phenomenon, and seeing that is the point of this post.

When you post a resting limit order, you are effectively running a one-sided auction. You've said "I will trade at \$100." Anyone in the world can take that offer. Who takes it? The person for whom \$100 is the most attractive — which, in a common-value world, means the person whose information says your \$100 is *wrong in their favor*. You posted a price; the market "won" the right to trade against you precisely when winning was bad for you. A resting order is a standing invitation, and the invitations that get accepted fastest are disproportionately the ones you should regret.

This is why **a fast fill is bad news.** Let's make the mechanism precise.

### Why instant fills are toxic

Picture two reasons your buy order at \$100 might fill. (See the contrast in the cover figure above: the toxic path versus the benign path.)

1. **Benign fill.** Someone needed to sell — an index fund rebalancing, a retiree raising cash, a fund meeting a redemption. They don't have a view; they just need liquidity. They hit your bid, you buy at \$100, and the price wanders around \$100 afterward because nothing fundamental happened. You earned the spread for providing liquidity. Good trade.

2. **Toxic fill.** Someone *knows* the stock is about to fall — they saw a seller's algo working, a news wire, an options-flow signal. They dump into your bid before the price drops. You buy at \$100; seconds later it's \$99.50, then \$99. You didn't earn the spread; you caught a falling knife. The counterparty's information became your loss.

The tell that separates them is *speed and continuation*. A benign seller trades a bit and stops; the price doesn't run. A toxic seller trades *fast* (they want in before the move) and the price keeps going *through* your level (because the information is real and the market is repricing). So an order that fills in milliseconds and is followed by more trading in the same direction is, on average, the toxic kind. The market microstructure name for losing money to better-informed counterparties on your resting quotes is, again, **adverse selection** — the exact same idea as Akerlof's lemons, now applied to who chooses to trade against your price.

Why is *continuation* — the price moving through your level — the sharpest tell? Because an informed trader has a finite, valuable piece of news, and they want to trade on it before it becomes public and the price moves without them. So they trade urgently and they trade *size*, taking liquidity wherever they can, including through your resting order. The trade footprint of real information is a burst of same-direction volume that walks the price. A benign liquidity need, by contrast, is *price-insensitive in the other direction*: a fund raising cash will sell into strength and ease off into weakness, so its trading tends to *dampen* moves rather than extend them. Continuation says "the person who traded with me knew the price was wrong, and the market is now agreeing with them." Mean-reversion says "the person who traded with me just needed to transact, and the price was fine."

There is a measurable version of this that desks track constantly: **markout**, the profit or loss on a fill measured a fixed time later (one second, ten seconds, a minute). If your fills, marked out a few seconds later, are systematically *underwater* — you bought and the price is lower, you sold and it's higher — your flow is toxic and you are being adversely selected. If your fills mark out roughly flat or slightly positive, your flow is benign and you're earning the spread. Markout is the empirical fingerprint of adverse selection, and a market maker whose markouts go red will widen, skew, or pull their quotes within milliseconds. You can do a crude version of the same thing by hand: after a passive fill, glance at the price ten seconds later. Consistently worse means you're the one getting picked.

#### Worked example: the spread you earned vs the adverse-selection cost

Say you make a market in a stock: you post a bid at \$99.95 and an ask at \$100.05, a 10-cent spread, mid at \$100.00. Trade 1,000 shares each way and the spread earns you about \$0.10 × 1,000 = **\$100** if both sides are benign noise traders who don't move the price.

Now suppose 30% of the volume hitting your quotes is informed — it trades against you right before a 30-cent move. On those informed fills you lose roughly \$0.30 per share minus the half-spread you collected (\$0.05), so about \$0.25 of adverse selection per informed share. If 300 of your 1,000 shares-per-side are informed, that's 300 × \$0.25 = **\$75** of losses, eating most of the \$100 you earned from the spread. Earn \$100 in spread, give back \$75 to the informed — your net is a thin \$25, and if the informed fraction or the move size ticks up, you go negative. That razor-thin margin is the whole reason dealers obsess over flow toxicity.

The intuition: the spread is your fee for providing liquidity, and adverse selection is the bill the informed traders run up against that fee.

### Toxic vs benign flow, side by side

The single most useful classification a liquidity provider makes is "is this flow toxic or benign?" because it determines whether resting in the market is profitable. The grid below lays out the two types along the dimensions that distinguish them: *why* they trade, and *how* the fill behaves.

![Toxic informed flow versus benign uninformed flow across why they trade and how the fill behaves](/imgs/blogs/adverse-selection-and-the-winners-curse-why-a-fast-fill-is-bad-news-4.png)

Toxic flow trades because it *knows* something — a signal, a speed advantage, an impending order — and it fills you fast and keeps going. Benign flow trades because it *needs* something — to rebalance, to fund a redemption, to hedge — and it fills you slowly while the price mean-reverts. You make money from the right column and lose it to the left. The entire craft of modern market-making is building systems to estimate, in real time, which column a given fill belongs to, and to widen, pull, or skew quotes when the flow looks toxic.

This is also why retail order flow is *valuable* and gets paid for. A retail investor buying 50 shares of an index ETF for their 401(k) is the purest benign flow — almost never informed about short-term moves. Wholesalers pay brokers for the right to trade against that flow precisely because it is the safe, right-column kind. The flow that *doesn't* get paid for, that exchanges and dealers are wary of, is the fast, aggressive, informed kind. When you trade, it's worth knowing which one you look like — and if you're a slow, uninformed long-term investor, the comforting news is that you are the flow everyone *wants* to trade against, which means you generally get filled at fair prices.

#### Worked example: reading your fills through markout

You're providing liquidity, posting a bid at \$50.00 in a stock with a fair value around \$50.02. Over a morning you get 20 passive fills. You check the price 10 seconds after each one.

On 14 of the fills, the price 10 seconds later is around \$50.03 — you bought at \$50.00 and it drifted up a few cents. Those 14 are benign: you earned roughly \$0.03 of markout each, about \$0.42 per share-lot in your favor. On the other 6 fills, the price 10 seconds later is \$49.85 — you bought at \$50.00 and it dropped 15 cents. Those 6 are toxic: you lost \$0.15 of markout each, about \$0.90 against you.

Net across 20 fills: $14 \times (+0.03) + 6 \times (-0.15) = 0.42 - 0.90 = -\$0.48$ per lot. Even though 70% of your fills were good, the 30% toxic minority — bigger losers each — turned the whole book negative. This is the adverse-selection arithmetic in miniature: it doesn't take a majority of informed flow to bleed you, just a toxic minority whose losses outweigh the spread you earn from everyone else.

The intuition: count your fills by *markout*, not by how good the fill *felt*, because the toxic minority is where your edge quietly dies.

### The winner's curse, restated as a trading rule

Step back and notice that the order-flow story and the auction story are the *same* selection effect wearing two outfits. In the auction, you win when your estimate was the most optimistic — so winning is adverse. In order flow, your resting bid gets hit fastest exactly when a seller's information says your bid is too high — so getting filled is adverse. In both cases, *the outcome you achieved (the win, the fill) is correlated with the state of the world being against you.* That correlation is the whole subject of this post.

So the winner's-curse discipline translates directly: just as an auction bidder shaves below their estimate to survive winning, a liquidity provider posts a quote *worse than their fair-value estimate* to survive getting filled. The shave in the auction and the spread on the quote are the *same adjustment* — a deliberate margin set so that, conditional on the trade happening, you break even on average rather than systematically losing to the information that triggered it. A market maker who quotes exactly at their fair value, with no spread, is the auction bidder who bids exactly their signal: they will get filled precisely when they're wrong, and bleed.

## How market makers defend: widening the spread

A market maker can't read minds. They can't label each incoming order "informed" or "uninformed" before filling it. So they do the only thing they can: they charge *everyone* a price for the risk that *some* of the flow is informed. That price is the **bid-ask spread** — the gap between the price they'll buy at (bid) and sell at (ask).

The logic runs like the pipeline below. Start with a tight quote. The informed traders pick off whichever side is mispriced and win, so the dealer loses on every informed fill — that's the adverse-selection cost. To survive, the dealer widens the spread until the extra money collected from *everyone* covers the losses to the *informed few*. The uninformed traders end up subsidizing the dealer's losses to the informed, and the dealer breaks even. The more toxic the flow, the wider the quote.

![A market maker widening its spread step by step to cover adverse-selection losses to informed traders](/imgs/blogs/adverse-selection-and-the-winners-curse-why-a-fast-fill-is-bad-news-5.png)

This is the heart of why spreads exist at all, and why they widen in exactly the moments you'd want them tight. Spreads blow out around earnings releases, economic data prints, and during fast markets — not because dealers are greedy in those moments, but because the *fraction of informed flow spikes*, and the spread has to rise to cover the higher adverse-selection cost. A wide spread is a dealer telling you "I think a lot of the people trading right now know more than I do."

#### Worked example: how wide must the spread be?

There's a clean toy version of this, and the kit's `data_gametheory.glosten_milgrom` function computes it. Suppose a stock is worth either \$110 (good news) or \$90 (bad news), each equally likely, so the prior mid is \$100. A fraction of the traders are informed (they know which) and the rest are noise traders (they buy or sell at random).

The dealer sets the ask so that, *given* someone is buying, the expected value covers the chance the buyer is an informed trader who knows it's the \$110 outcome. Run the numbers:

- If **10%** of flow is informed, the model gives bid \$99, ask \$101 — a **\$2** spread.
- If **30%** is informed, bid \$97, ask \$103 — a **\$6** spread.
- If **50%** is informed, bid \$95, ask \$105 — a **\$10** spread.

The spread is a direct, increasing function of the informed fraction. Triple the toxicity (10% to 30%) and the spread triples (\$2 to \$6). The dealer isn't forecasting the stock; they're pricing the *probability that you know something they don't*. The full formal version of this — the Glosten-Milgrom model, where the dealer Bayesian-updates the price after every single trade — gets its own post in the next track of this series; here it's enough to see that the spread *is* the adverse-selection premium.

The one-sentence intuition: the spread you pay is the dealer's insurance premium against the informed traders hiding in the crowd with you.

## Common misconceptions

**"A fast fill means I got a great price."** This is the single most expensive intuition in trading, and it's exactly backward for a resting (passive) order. If you posted a limit order and it filled instantly while the market kept moving through your level, you most likely got *picked off* by someone informed. The fast fill is evidence the price was *wrong in their favor*. (For an *aggressive* order — you crossing the spread to take liquidity — a fast fill just means there was size resting; that's different. The toxic-fast-fill warning is about your *passive* orders.) Speed of fill on a resting order is a toxicity signal, not a quality signal.

**"Winning an auction means I valued the item correctly."** In a common-value setting, winning means you valued it *most*, which means you most likely valued it *too high*. Your win is selected from the right tail of everyone's estimates. The correct response to winning a competitive bid for an uncertain common value is a flicker of worry, not satisfaction — and the bigger the field you beat, the more worried you should be, because beating more rivals means clearing a higher bar of optimism.

**"More bidders is always better."** True for the *seller*, false for *you as a buyer* in a common-value auction. Every extra rival you have to outbid means winning requires you to have been more optimistic, which means a bigger overpayment. The chart of overpayment vs number of bidders climbs the whole way. If you're bidding against a crowd for something with a shared, uncertain value — a hot IPO allocation, a competitive takeover, a contested block — the crowd is a reason to shade harder, not bid up.

**"Tight spreads are always a sign of a healthy, cheap market."** Tight spreads mean dealers think flow is mostly benign right now. The instant flow turns toxic — a data surprise, a fast move — spreads blow out, and the "cheap" market vanishes exactly when you most want to trade. A tight spread is cheap liquidity *conditional on nothing happening*; it is not a promise. The spread is a real-time toxicity gauge, not a fixed feature of the stock.

**"If I'm a small retail trader, none of this applies to me."** It applies, but mostly in your favor. As a slow, uninformed trader you *are* the benign flow that dealers want, so you generally get filled at fair prices and your market orders rarely move the price against you. The trap is only when you start trading *fast on news* or in *illiquid names* — then you either become the informed flow that dealers defend against (wide spreads) or you get adversely selected by faster players. Trade slowly and patiently and adverse selection mostly works *for* you.

## How it shows up in real markets

### The flash crash and the toxic-flow morning (May 6, 2010)

The flash crash is the textbook case of adverse selection scaling up to a market-wide event. On May 6, 2010, the S&P 500 E-mini futures and US equities fell roughly 9% and recovered most of it within about 36 minutes. The joint SEC-CFTC report later traced the trigger to a large automated sell program in the E-mini. As the selling hit, market makers' fills became overwhelmingly toxic — every fill was followed by more selling — so liquidity providers did exactly what the model predicts: they widened, then pulled their quotes entirely rather than keep getting adversely selected. Some stocks printed at a penny, others at \$100,000, as bids evaporated. The lesson practitioners took away was precisely the one in this post: when your fills suddenly become fast and one-directional, the rational response is to *stop providing liquidity*, and when everyone does that at once, the market gaps.

There's a feedback loop worth naming here, because it's how a normal day becomes a crash. Each individual market maker, seeing its own markouts go red, rationally widens and then pulls — protecting itself from adverse selection. But when every market maker does that simultaneously, the visible liquidity that *was* absorbing the sell program vanishes, so the next slice of selling moves the price even more, which makes the *next* fills even more toxic, which makes the remaining liquidity providers pull *faster*. The defense each player takes against being adversely selected is individually correct and collectively catastrophic. This is why "stub quotes" (placeholder orders at absurd prices like a penny) got hit during the crash: they were the only bids left after everyone with real capital had stepped away from flow that had become pure toxicity. Circuit breakers — automatic trading halts — were strengthened afterward precisely to interrupt this loop and give the informed/uninformed mix time to normalize.

### The "lemons" in the 2008 mortgage market

Akerlof's used-car logic played out at planetary scale in 2007-2008. Mortgage-backed securities and CDOs were sold on the assumption that buyers couldn't easily tell a good pool of loans from a bad one — the quality was opaque, and the *sellers* (originators packaging the loans) knew more than the buyers. As losses surfaced in 2007, buyers couldn't distinguish solid pools from toxic ones, so — exactly like the used-car market — they discounted *everything* and the market for these securities froze. Trading in many structured products simply stopped: not because every bond was bad, but because nobody could tell, and the willingness to sell had become a bad signal. That freeze, an adverse-selection unraveling, was a core mechanism of the credit crisis.

### Spreads widening around scheduled news

You can watch the adverse-selection premium in the spread on any given day. In the seconds before a major US data release — the monthly jobs report (nonfarm payrolls), CPI inflation, or an FOMC interest-rate decision — bid-ask spreads in Treasury futures, equity index futures, and FX visibly widen, and depth at the top of the book thins out. The reason is pure Glosten-Milgrom: around the release, the *fraction of informed (or about-to-be-informed) flow spikes*, so dealers raise the spread to cover the higher adverse-selection cost, and many pull quotes entirely across the release itself. The same names trade at a 1-tick spread at 2pm and a 5-tick spread at 8:29am before an 8:30am print. The spread is the toxicity gauge, ticking up in real time.

### Payment for order flow and the value of benign retail

The economics of US equity wholesaling are adverse selection made visible. Wholesalers like Citadel Securities and Virtu pay retail brokers for the right to execute their customers' orders — *payment for order flow*. Why pay for the privilege of taking the other side of someone's trade? Because retail flow is the cleanest benign, right-column flow there is: a person buying 30 shares of an ETF for their retirement account is almost never informed about the next 30 seconds. Internalizing that flow lets the wholesaler capture the spread with very little adverse-selection risk, and they share some of that value back with the broker. The flip side: institutional flow, which is more likely to be informed, doesn't get paid for and is treated with much more caution. The price of order flow is essentially a market price for *how benign it is*.

### The winner's curse in spectrum and Treasury auctions

The winner's curse is not a metaphor in the auction world — bidders explicitly shade for it. The phrase itself was coined by petroleum engineers at Atlantic Richfield in 1971, who noticed that the companies winning offshore oil-lease auctions were consistently disappointed by the actual yields: the winner was reliably the firm that had been most optimistic about the geology, and most optimistic meant most wrong. In government spectrum auctions (the FCC's sales of wireless frequencies), telecom companies bidding for licenses worth a common, uncertain amount have learned to shade their bids below their raw estimates, because the firm that wins is the one that most overestimated demand. Early bidders who didn't shade — in some 1990s spectrum and oil-lease auctions — won blocks and then took writedowns, the textbook curse. The US Treasury moved its bill and note auctions to a *uniform-price* (single-price) format partly to reduce winner's-curse-driven bid shading: when every winner pays the same clearing price rather than their own bid, the incentive to shave is smaller, which encourages more aggressive bidding and tighter results for the taxpayer. The full IPO-and-auction version of the winner's curse — including why IPOs are systematically underpriced as compensation for it — gets its own dedicated post later in this series' Auctions track.

### M&A and the curse of the winning acquirer

Corporate takeovers are common-value auctions with billions of dollars at stake, and the winner's curse shows up as a well-documented pattern: the *acquirer's* stock often falls when a deal is announced, especially after a competitive bidding contest. The logic is exactly the model. Several companies estimate what a target is worth — a common value, since the target's future cash flows are the same regardless of who buys it — and the one that wins is the one that bid highest, which is the one that most overestimated the synergies. Studies of merger waves repeatedly find that acquirers who win contested auctions tend to overpay, and that the more bidders a target attracts, the worse the eventual return to the winner. Disciplined acquirers know this and set a *walk-away price* below their estimate — a corporate version of the shave — and are willing to lose deals to less disciplined rivals who will later regret winning. "We were outbid" is, in a common-value contest, frequently the words of the smarter party.

### Block trades and the careful seller

When a fund needs to sell a large block of stock, the entire negotiation is a dance around adverse selection. If the seller dumps it fast and aggressively, the dealer on the other side assumes the seller is informed (why else the rush?) and quotes a punishing price or refuses. So sophisticated sellers deliberately trade *slowly and patiently*, breaking the order into small pieces (an algorithm like VWAP or a "TWAP" that spreads execution over hours), precisely to *look benign* — to signal "I'm just a liquidity need, not an informed trader." The art of execution is, in large part, the art of not looking like toxic flow. The flip side for the dealer: a counterparty in an unusual hurry is a counterparty to fear.

### Insurance and the original adverse-selection market

The phrase "adverse selection" actually comes from insurance, and the insurance case makes the mechanism vivid. An insurer offering health or life cover at a single average price attracts a biased pool: the people most eager to buy generous coverage are disproportionately the ones who privately know they're sick or high-risk. The healthy, who know they're healthy, find the average price a bad deal and decline. So the pool that *buys* is sicker than the population average, the insurer's payouts exceed what the average price assumed, and they must raise the price — which drives out the next-healthiest tier, worsening the pool again. It's the lemons spiral in a different costume: the willingness to buy insurance is correlated with bad news (for the insurer). Insurers fight back with the same tools markets do — medical exams (a signal, like a warranty), risk-based pricing (a wider, customer-specific spread), and mandates that force the healthy into the pool (regulating away the self-selection). Every defense maps onto something a market maker does.

### Dark pools and the toxicity of the venue

Modern equity markets are fragmented across dozens of venues, including *dark pools* — private trading venues that don't display quotes. Institutions use them to trade large size without showing their hand, which sounds like pure benign-flow heaven. But adverse selection follows the information, not the venue: if a dark pool develops a reputation for hosting informed, predatory flow — players who detect a big resting order and trade against it — then liquidity providers mark out badly there and pull back, and the pool's quality degrades. Operators police this by measuring participant markouts and segmenting or ejecting toxic participants, exactly the way a card room bars a cheater. The result is a quiet, ongoing sorting of venues by flow toxicity: the "cleanest" pools, where markouts are flat, attract the most liquidity, while toxic venues spiral toward thin books and wide effective spreads. The reputation of *where* you trade is itself a signal about *who* you'll trade against.

## The playbook: how to play it

This series always ends on *so how do you play this?* Here is the operational version.

**Know which flow you are.** If you trade slowly, in liquid names, on no special information, you are benign right-column flow — adverse selection works *for* you, you get fair fills, and you should mostly ignore the toxicity machinery except to be glad of it. If you trade fast, on news, in thin names, you are either the informed flow (and dealers will defend against you with wide spreads) or you are the one getting adversely selected by someone faster. Be honest about which you are, because it determines whether speed helps or hurts you.

**Read your fills, don't just take them.** After a passive (resting limit) fill, watch the next few seconds. If the price keeps moving *through* your level in the direction that hurts you, that fill was probably toxic — you were on the wrong side of information, and you should expect more of the same if you keep quoting there. If the price mean-reverts back toward your level, the fill was benign and your quote is well-placed. A fill is a piece of data about your counterparty; the fast, continuing ones are the warnings.

**Treat the spread as a toxicity gauge.** When spreads in your market are wide and depth is thin, the dealers are telling you the informed fraction is high — typically around scheduled news, fast moves, or in illiquid names. That is a worse time to demand liquidity (you pay the premium) and a worse time to provide it (you get adversely selected). When spreads are tight, flow is benign — fine for patient trading, but don't mistake it for a guarantee, because it can flip in an instant.

**In any common-value contest, shave your bid.** Whenever you find yourself bidding for something whose value is uncertain and shared — a competitive block, a hot allocation, a takeover, anything where many parties are estimating the same number — bid your honest estimate *minus* the winner's-curse correction. The shave grows with the number of rivals and with your own uncertainty: roughly $w \cdot \frac{n-1}{n+1}$ off your estimate. Concretely, against ~10 rivals with substantial uncertainty, shading 15-20% off your raw estimate is not timid; it's the break-even bid. If you win every contest you enter, you are shading too little.

**Reduce your noise before you raise your bid.** The cleanest way to bid more aggressively without getting cursed is to *sharpen your estimate* — better data, better models, a genuine informational edge. A bidder with half the noise suffers half the curse and can outbid you safely. If you don't have an edge on the value, your only protection is the shave; never confuse "I want this badly" with "this is cheap."

**Don't provide liquidity in the moments flow turns toxic.** If you make markets, even informally — posting limit orders to capture the spread, selling covered calls, providing liquidity in a thin name — the single most important risk-management move is to *step back when the informed fraction spikes*. Concretely: pull or widen your resting orders ahead of scheduled news (the jobs report, CPI, an FOMC decision, an earnings release), in the first and last few minutes of the session, and whenever you see a fast one-directional move. These are precisely the windows where the share of informed flow jumps and your markouts go red. The market makers who survive are not the ones with the best forecast; they're the ones who *stop quoting* when they can tell the people trading against them probably know more.

**Make yourself look benign when you need liquidity.** The mirror image: when *you* are the one who needs to trade size, your job is to *not* look informed, because looking informed gets you a worse price. Trade patiently, split the order across time, avoid the urgent all-at-once execution that screams "I know something." If you genuinely are uninformed (you're rebalancing, raising cash, hedging), advertising that — by trading slowly and predictably — gets you treated as the benign flow you are, and benign flow gets fair prices. The trader who panics and dumps a position into a thin book pays twice: once for the move they cause, and once for the toxicity premium dealers charge anything in a hurry.

**The invalidation.** This whole framework assumes your counterparties might be informed. In a genuinely private-value situation — you're buying a house *to live in*, or an asset whose value to you is personal and idiosyncratic — the winner's curse mostly doesn't apply, and shading is just leaving the thing you wanted on the table. The skill is telling common-value situations (shade, fear the fast fill) from private-value ones (bid your value, a fast yes is fine). Misclassify a private-value deal as common-value and you'll be too timid; misclassify a common-value deal as private-value and you'll be the one taking the writedown.

The thread back to the series spine: a trade is a strategic interaction, not a bet against nature. The other side chose to deal with you for a reason, and the faster and more eagerly they chose it, the more that reason is likely to be at your expense. Reading that — turning "I got my fill" into "what did agreeing to this tell me about the person who agreed?" — is the edge. The next post formalizes the dealer's side of it with the Glosten-Milgrom model; this one was about learning to flinch at a fast fill.

## Further reading & cross-links

- [The Trade Is a Game: Why Markets Are Strategic, Not Random](/blog/trading/game-theory/the-trade-is-a-game-why-markets-are-strategic-not-random) — the series opener on why every trade has a specific, reasoning counterparty rather than an indifferent nature, the foundation this post builds on.
- [Zero-Sum, Positive-Sum, and the House: Where Trading Profits Come From](/blog/trading/game-theory/zero-sum-positive-sum-and-the-house-where-trading-profits-come-from) — where your profits actually come from, and why benign-flow counterparties are the ones you want.
- [How an Options Market Maker Thinks: The Other Side of Your Trade](/blog/trading/options-volatility/how-an-options-market-maker-thinks-the-other-side-of-your-trade) — the dealer's full view of adverse selection and spread-setting, the natural deep-dive on the defending side of this post.
- [The SIG / Susquehanna Playbook: Poker, Game Theory, and EV](/blog/trading/quant-careers/sig-susquehanna-playbook-poker-game-theory-and-ev) — how a top prop firm trains the exact instinct of pricing the information in your counterparty's eagerness.

*Educational, not financial advice. The models here are deliberately simplified to build intuition; real markets add depth, latency, fees, and adversaries that don't read the textbook.*
