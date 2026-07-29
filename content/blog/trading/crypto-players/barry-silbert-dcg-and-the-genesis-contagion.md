---
title: "Barry Silbert, DCG, and the Genesis Contagion: How One Owner's Plumbing Became Everyone's Problem"
date: "2026-07-29"
publishDate: "2026-07-29"
description: "A from-zero anatomy of Digital Currency Group — the 2% trust fee, the six-month lockup that made the GBTC premium trade, the closed-end arithmetic that guaranteed a discount, and how a single borrower's default climbed the org chart into a $1.1 billion promissory note and 340,000 frozen retail accounts."
tags: ["crypto", "digital-currency-group", "grayscale", "gbtc", "genesis", "gemini-earn", "barry-silbert", "closed-end-fund", "related-party", "crypto-lending", "contagion", "crypto-players"]
category: "trading"
subcategory: "Crypto Players"
author: "Hiep Tran"
featured: true
readTime: 59
---

> [!important]
> **TL;DR** — Digital Currency Group owned the company that sponsored the bitcoin trust, the broker-dealer that was the trust's *only* share creator, and the lender that borrowed 340,000 retail savers' coins. When one borrower defaulted, the loss had a private road into every other pocket.
>
> - **GBTC could inhale but not exhale.** Its redemption program was suspended on **28 October 2014**; new shares carried a **six-month Rule 144 lockup**. The trust's own 10-K states the consequence in plain words: "there is **no arbitrage mechanism**" to keep the share price near the value of the bitcoin inside.
> - That missing mechanism let the price roam. Across the entire closed-end era — **5 May 2015 to 10 January 2024** — GBTC's maximum premium to NAV was **+142%** and its maximum discount was **−49%**, and the shares closed at a discount on **725 separate days** (Grayscale Bitcoin Trust ETF 10-K for FY2025).
> - **The fee was paid in bitcoin.** A **2.0%** annual sponsor's fee — the rate from inception until 10 January 2024, cut to **1.5%** the next day (GBTC 10-K) — levied in the asset itself, means each share holds strictly less bitcoin every year.
> - In **mid-June 2022** Three Arrows Capital defaulted. The SEC's later order describes an **"approximately \$1 billion loss"** at Genesis, and a **promissory note with a 10-year term** from parent to subsidiary. The New York Attorney General puts the note at **\$1.1 billion at 1% interest, executed 30 June 2022 and payable in a decade** — and alleges no principal or interest was ever paid on it.
> - **The number to remember: zero.** Per the SEC's January 2025 order, DCG "had in fact not transferred any capital to Genesis" when it was described as having ensured Genesis had adequate capital to operate. A note is a promise; a promise is not cash.

For years there was a trade in crypto that looked like a law of physics. You handed bitcoin to a trust, waited six months, sold the shares you got back, and collected the gap — not from bitcoin going up, but from the *wrapper* being worth more than its contents. That gap averaged **37%** across the trust's life and at its peak reached **142%** (Grayscale Bitcoin Trust 10-K). Hedge funds levered into it. Lenders financed it. For years it worked.

Then it stopped working, and the same structure ran in reverse. By December 2022, that wrapper traded at nearly half off. The funds that had borrowed to feed the machine owed back real bitcoin and held shares worth a fraction of it. One of those funds, Three Arrows Capital, defaulted on a margin call. The lender it defaulted to was called Genesis. Genesis' parent was called Digital Currency Group. And DCG also owned the company that sponsored the trust at the center of the whole trade.

![The DCG stack: one holding company sat above the trust's sponsor, the trust's only authorized participant, and the lender that borrowed retail savers' coins.](/imgs/blogs/barry-silbert-dcg-and-the-genesis-contagion-1.webp)

The diagram above is the mental model for everything that follows. It is not a picture of fraud — it is a picture of *plumbing*. One owner, several businesses, and pipes between them that no outside depositor could see. This post takes that structure apart from zero: what a holding company is, what a trust is, why a closed-end fund with no redemption right **must** eventually trade at a discount, how a lender's spread works, and how a single borrower's default climbed the org chart until it landed on 340,000 people who thought they had a savings account.

A note on how this is written, because the fairness of it matters as much as the mechanics.

Matters of public record — the bankruptcy filing, the regulatory actions, the settlement amounts, the dates — are stated as fact with their source. Everything contested is framed as *alleged* or *per the complaint*, with the source and date.

Three things are worth stating plainly at the outset. First, **Barry Silbert has been sued civilly; he has not been criminally charged.** The New York Attorney General's case is a civil action, and nothing in this post should be read as suggesting otherwise. Second, several matters here settled **without any admission or denial of wrongdoing**, and where that is true this post says so every time. Third, the Attorney General's fraud allegations against DCG and Silbert are **unproven and contested** — DCG has not settled those claims. DCG and Silbert are widely reported to have rejected the allegations publicly, but I was not able to source a directly quotable response from either while writing this, so rather than paraphrase a denial I have not read, I simply note that the claims are disputed and untested.

This is an explanation of a mechanism, not a verdict about a person, and it is educational material rather than investment advice.

## Foundations: the building blocks

If you have never read a corporate filing, this section is the whole vocabulary you need. If you have, skim to the next heading.

### A company that owns other companies

A **holding company** is a company whose business is owning other companies. It usually makes nothing and sells nothing itself. Its assets are the shares of its **subsidiaries** — the operating businesses underneath it.

Think of a landlord who owns five restaurants through five separate companies. The landlord is the holding company. Each restaurant is a subsidiary. Why bother with the structure? Two reasons. First, **limited liability**: if one restaurant is sued into oblivion, the creditors of that restaurant can generally reach only that restaurant's assets, not the other four. That legal wall between entities is what people mean by the *corporate veil*. Second, flexibility: you can sell one restaurant, or raise money against one, without touching the others.

Digital Currency Group is a holding company. Grayscale Bitcoin Trust's annual report describes its sponsor, Grayscale Investments, LLC, as "a Delaware limited liability company formed on May 29, 2013 and a wholly owned subsidiary of Digital Currency Group, Inc." — and adds the crucial legal point that "DCG, the sole member of the Sponsor, is not responsible for the debts, obligations and liabilities of the Sponsor solely by reason of being the sole member."

Hold that sentence. It cuts both ways. The wall that stops a subsidiary's creditors reaching the parent also means the parent is not *obliged* to rescue the subsidiary. When we get to a lender with a hole in it, that asymmetry is the whole game.

### A transaction between two pockets of the same trousers

A **related-party transaction** is a deal between two entities controlled by the same people — a parent lending to its subsidiary, or two sister companies contracting with each other.

Related-party deals are legal, common, and often mundane. The reason regulators obsess over them is that the ordinary discipline of a negotiation is missing. When you and I negotiate a loan, we each push for our own side, and the interest rate that comes out reflects the real risk. When a parent lends to its own subsidiary, nobody is on the other side of the table. The rate can be whatever the group finds convenient. The term can be whatever is convenient. And the asset that appears on the subsidiary's balance sheet is worth whatever the parent's promise is worth — a judgment the subsidiary's own creditors are in no position to make.

GBTC's 2023 annual report flags exactly this in its risk factors, noting the "absence of arm's-length negotiation with respect to certain terms of the Trust, and, where applicable, there has been no independent due diligence conducted with respect to the Trust."

### A fund, its NAV, and its price

A **fund** is a pot of assets that many people own slices of. Each slice is a **share**.

**NAV** — *net asset value* — is the value of everything in the pot, minus what the pot owes, divided by the number of shares. It answers: "what is one share actually worth in underlying stuff?"

The **market price** is a different number entirely: it is what someone will pay you for the share today. NAV is about contents. Price is about demand.

For most funds these two numbers stay glued together, and the glue has a name.

### The glue: redemption, and why it is everything

There are two shapes a fund can take.

An **open-end fund** (an ETF is the familiar version) lets big institutional traders called **authorized participants** create and destroy shares on demand at NAV. If the share price drifts *above* NAV, an authorized participant deposits the underlying assets, receives new shares, sells them at the higher price, and pockets the difference — which pushes the price back down. If the price drifts *below* NAV, they buy cheap shares, redeem them for the underlying assets, sell those, and pocket the difference — which pushes the price back up. This two-way trade is **arbitrage**: a near-riskless profit taken by exploiting a gap between two prices of the same thing. Its side effect is that the gap closes. The redemption right is the rope that ties price to NAV.

A **closed-end fund** has a fixed number of shares and no ongoing redemption. You cannot hand your share back for the underlying assets. If you want out, you sell to another investor at whatever price they will pay. The rope is gone.

![Without a redemption right there is no rope: an ETF's price is dragged back to NAV every day, while a closed-end trust's price is free to roam.](/imgs/blogs/barry-silbert-dcg-and-the-genesis-contagion-2.webp)

This single structural difference generates most of what happened to DCG. We will do the arithmetic properly in a moment, because it is the most important piece of arithmetic in the story.

### A lender that stands in the middle

A **prime broker** in traditional markets is the firm that lends money and securities to hedge funds, holds their assets, and clears their trades. Crypto grew its own version. Genesis was that: it borrowed crypto from people who had it and lent it to trading firms who wanted it, keeping the difference.

The difference is called the **spread**. If you borrow at 5% and lend at 9%, your spread is four percentage points. Note the shape of that business: the profit arrives in thin annual slices, and the loss — if a borrower fails to pay — arrives all at once and can be many years of profit in a single afternoon.

The buffer that absorbs such a loss is the lender's own **equity** — the money the owners put in, which by law takes losses before the depositors do. A bank with \$100 of equity against \$1,000 of loans can absorb a \$100 loss before its depositors feel anything. Above that, the depositors start losing.

### The players in this story

| Entity | What it did | Who owned it |
| --- | --- | --- |
| **Digital Currency Group (DCG)** | Holding company, Stamford CT; founded 2015 | Barry Silbert, founder and CEO |
| **Grayscale Investments** | **Sponsor** of GBTC; collected the 2.0% fee | Wholly owned by DCG |
| **GBTC** (Grayscale Bitcoin Trust) | The trust that held the bitcoin | Sponsored by Grayscale; owned by its shareholders |
| **Genesis Global Trading** | Broker-dealer; GBTC's **sole authorized participant** until 3 Oct 2022 | Wholly owned by DCG |
| **Genesis Global Capital** | The lender; borrower of Gemini Earn deposits | DCG subsidiary via Genesis Global Holdco |
| **CoinDesk** | Media outlet covering the industry | Owned by DCG until 2023 |
| **Gemini Trust Company** | Exchange; ran Gemini Earn as **agent** | Cameron and Tyler Winklevoss — *not* DCG |
| **Three Arrows Capital (3AC)** | Hedge fund; the borrower that defaulted | Su Zhu and Kyle Davies — *not* DCG |

Two entities in that table are not DCG companies, and that matters: Gemini and 3AC were genuinely independent counterparties. The entanglement being described is inside the DCG column.

## The man and the machine

Barry Silbert's own biography, as it appears in GBTC's annual report, is unusually useful because it is a filed document rather than a profile: "Prior to leading DCG, Mr. Silbert was the founder and CEO of SecondMarket, a technology company that was acquired by Nasdaq." Before that, "Mr. Silbert worked as an investment banker."

SecondMarket is the key to understanding what came next. It was a marketplace for *illiquid* assets — shares in private companies, restricted stock, things that were hard to sell because no exchange listed them. Silbert's professional formation was in the business of building a venue where something previously untradeable could change hands.

The Grayscale trust is that idea applied to bitcoin. In 2013, a US institution that wanted bitcoin exposure faced a wall: custody was hard, compliance was harder, and no spot bitcoin ETF existed. GBTC's answer was to put bitcoin inside a security with a ticker that a normal brokerage account could hold. The trust's first creation basket dates to **25 September 2013**, per the 10-K.

The same filing describes DCG's scope: founded in 2015, "backing more than 150 companies across 30 countries, including Coinbase, Ripple, and Chainalysis," and investing "directly in digital currencies and other digital assets."

By November 2021, DCG raised capital in a secondary share sale that press reports at the time valued the group at more than **\$10 billion** (CNBC and CoinDesk, 1 November 2021). Treat that as a reported private-market mark rather than an audited figure — private valuations are negotiated, not observed.

The structure that valuation sat on top of is what the rest of this post is about.

## The trust that could inhale but not exhale

GBTC had a one-way valve, and it was not an accident of design — it was the residue of a regulatory problem.

### The valve, and how it got stuck

The 10-K records the moment: "Effective October 28, 2014, the Trust suspended its redemption program, in which shareholders were permitted to request the redemption of their Shares through Genesis, the sole Authorized Participant at the time out of concern that the redemption program was in violation of Regulation M under the Exchange Act, resulting in a settlement reached with the SEC."

Unpack that. **Regulation M** restricts an issuer from buying back its own securities while it is distributing them — the concern being that simultaneous buying and selling can be used to prop up a price. Grayscale's read was that running creations and redemptions at the same time might run afoul of it. So redemptions stopped in October 2014 and did not restart for more than nine years.

Note who is named in that sentence: **Genesis** was the sole authorized participant even in 2014. The lender that would later blow up was, from the very beginning, the single gateway through which shares of the trust were created and (until 2014) redeemed.

The other half of the valve is **Rule 144**, the SEC rule governing resale of securities that were sold privately rather than through a registered public offering. Per the 10-K: "Pursuant to Rule 144, a minimum six-month holding period applies to all Shares purchased from the Trust." Grayscale aggregated qualifying shares on a bi-weekly basis and had outside counsel instruct the transfer agent to strip the restrictive legends, at which point the shares could trade freely on OTCQX.

So: you could put bitcoin *in* and receive shares, but those shares were frozen for six months, and nobody could take bitcoin *out* at all. New tradable supply always lagged demand by half a year.

### The fee, and the fact that it was paid in bitcoin

The trust's main running cost is the **sponsor's fee**. The 10-K defines it precisely: "A fee, payable in Bitcoins, which accrues daily in U.S. dollars at an annual rate of 2.0% of the Digital Asset Holdings Fee Basis Amount of the Trust."

Three words there do a lot of work: *payable in Bitcoins*. The fee is calculated in dollars and then settled by taking bitcoin out of the trust. Grayscale earned that fee whether bitcoin went up, down, or sideways, and whether the shares traded at a premium or a discount. It is the most reliable revenue line in this entire story — and the reason the sponsor never had the same problem the lender did.

For the shareholder, it means something specific: your share holds a little less bitcoin every single day.

### Worked example 1: the creation-and-lockup round trip

Here is the trade that made GBTC famous, run end to end with the lockup and the fee included. Round numbers, illustrative.

![The premium round trip: deposit at NAV, wait out the Rule 144 lockup, sell above NAV — a trade that printed money only while the premium survived six months.](/imgs/blogs/barry-silbert-dcg-and-the-genesis-contagion-3.webp)

**Step 1 — deposit.** You hand Grayscale **\$1,000,000** of bitcoin and receive GBTC shares worth \$1,000,000 at NAV. In accounting terms nothing has happened yet: you swapped one asset for another of equal value.

**Step 2 — wait.** Six months, per Rule 144. You cannot sell. During this window the 2.0% annual fee accrues, and over half a year that is roughly 1.0% of your holding, taken in bitcoin:

$$\$1{,}000{,}000 \times (1 - 0.010) = \$990{,}000 \text{ of NAV remaining.}$$

**Step 3 — sell.** The legend comes off and you sell on OTCQX. The filings report an **average premium of 37%** across the whole closed-end era, 5 May 2015 to 10 January 2024. Apply that to your remaining NAV:

$$\$990{,}000 \times 1.37 = \$1{,}356{,}300.$$

**Step 4 — count.** You put in \$1,000,000 of bitcoin and took out \$1,356,300 of cash: a gain of **\$356,300**, or **35.6% in six months**, entirely separate from whatever bitcoin itself did.

Now the second-order version, because this is where the danger lives. Suppose you did not own the bitcoin — you **borrowed** it, as many funds did. You still owe the lender the same quantity of bitcoin back. Your obligation is denominated in *coins*; your asset is denominated in *shares*. Those two things are only equivalent while the premium holds. The moment the shares trade below NAV, you owe more coins than your shares can buy back, and no amount of patience fixes it, because the shares are locked.

**The one-sentence intuition:** the GBTC premium trade was never arbitrage — it was an unhedged six-month bet that the premium would still be there when the lockup expired.

### Worked example 2: the fee, compounded — and a check against the filing

The fee looks trivial and is not. Because it is taken in bitcoin, the right way to see it is as a decay in the *bitcoin content* of a share.

![The 2% fee is levied in bitcoin itself, so a share's bitcoin content decays on a fixed schedule regardless of price.](/imgs/blogs/barry-silbert-dcg-and-the-genesis-contagion-4.webp)

Start with an index of 1.000 bitcoin-per-share and take 2% a year:

$$B_n = B_0 \times (1 - 0.02)^n$$

where $B_n$ is the bitcoin per share after $n$ years and $B_0$ is the bitcoin per share when you bought.

| Years held | Bitcoin per share (index) | Cumulative fee drag |
| --- | --- | --- |
| 0 | 1.000 | 0.0% |
| 2 | 0.960 | 4.0% |
| 5 | 0.904 | 9.6% |
| 10 | 0.817 | **18.3%** |

Hold for a decade and roughly one bitcoin in five and a half has quietly left your share.

Now the satisfying part: we can check this against the trust's own disclosures and watch it come out right.

- The 10-K states each share in the **initial baskets** represented approximately **0.1 of a bitcoin**, with the first creation basket dated **25 September 2013**.
- On **26 January 2018** the trust "completed a 91-for-1 Share split."
- As of **31 December 2022**, "each Share represented approximately **0.0009** of one Bitcoin."

Adjust the starting point for the split:

$$\frac{0.1}{91} = 0.0010989 \text{ bitcoin per (post-split) share.}$$

Now run the fee from September 2013 to December 2022 — about **9.27 years** — at 2% a year:

$$0.0010989 \times (0.98)^{9.27} = 0.0010989 \times 0.8292 = 0.000911.$$

Which rounds to **0.0009** — exactly the figure the trust discloses. The fee alone explains the entire decline once you strip out the share split.

That last clause matters, and it is a trap worth naming. Looking only at "0.1 bitcoin per share in 2013, 0.0009 today" would suggest the fee ate 99% of the trust. It did not. The 10-K says the decrease is "primarily a result of the Share Split and, to a lesser degree, the periodic withdrawal of Bitcoin to pay the Sponsor's Fee." A share split changes the *unit*, not the value — like restating a price from dollars to cents. Only the second effect is a real cost.

**The one-sentence intuition:** a fee charged in the asset is a fixed, guaranteed, market-independent leak, and it is the only cash flow in this story that worked perfectly for nine straight years.

## Why a closed-end trust must eventually trade at a discount

This is the arithmetic the entire saga turns on, so we will build it slowly.

### The mechanism, stated plainly

In an ETF, price cannot stray far from NAV because straying *creates a profitable trade whose execution closes the gap*. That is the entire mechanism. Not regulation, not goodwill — a self-interested arbitrageur with a redemption right.

Remove the redemption right and ask what determines the price. Only two things: how many shares exist, and how badly people want them.

Now watch the supply side of GBTC. Every creation basket added shares permanently — nothing could ever remove them, because redemptions were suspended. Share count was a **ratchet**: it went up and never came down. And crucially, creations were most attractive *precisely when the premium was highest*, because the premium is what made the trade profitable. So a high premium mechanically manufactured the future supply that would destroy it. Every dollar of premium was an advertisement for more shares, arriving in six months.

Meanwhile demand was a fashion. GBTC's appeal rested on being the only convenient way for a US brokerage account to hold bitcoin. That moat drained: bitcoin futures ETFs arrived, competing trusts arrived, direct custody got easier, and eventually spot ETFs arrived.

Put the ratchet and the fashion together and the conclusion is not a forecast, it is a structural statement. Supply that only rises, demand that can fall, and no mechanism connecting either to NAV. Once inflows stop, there is nothing holding the price up — and nothing that *must* pull it back.

You do not have to take my word for it, because the trust itself said so. From the risk factors: "because of the holding period under Rule 144, the lack of an ongoing redemption program, and the Trust's ability to halt creations from time to time, **there is no arbitrage mechanism** to keep the value of the Shares closely linked to the Index Price and the Shares have historically traded at a substantial premium over, or substantial discount to, the Digital Asset Holdings per Share."

That disclosure was in the public filing the entire time.

<figure class="blog-anim">
<svg viewBox="0 0 780 380" role="img" aria-label="A schematic of GBTC's premium and discount over time: the share price runs above net asset value through 2020, crosses to a discount in early 2021 as new locked shares keep arriving and demand stops, reaches its deepest discount in late 2022, and snaps back to par when the trust converts to an ETF in January 2024" style="width:100%;height:auto;max-width:840px">
<style>
.pd-bg{fill:var(--surface,#f8fafc);stroke:var(--border,#d8dee6);stroke-width:1}
.pd-prem{fill:#b2f2bb;opacity:.55}
.pd-disc{fill:#ffc9c9;opacity:.55}
.pd-nav{stroke:var(--text-primary,#1f2937);stroke-width:2;stroke-dasharray:6 4}
.pd-line{fill:none;stroke:#1c7ed6;stroke-width:3.5;stroke-linejoin:round;stroke-linecap:round}
.pd-ax{stroke:var(--text-secondary,#64748b);stroke-width:1.5}
.pd-t{font:600 13px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.pd-s{font:500 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#64748b);text-anchor:middle}
.pd-k{font:700 15px ui-sans-serif,system-ui;text-anchor:middle}
.pd-sweep{stroke:#f59f00;stroke-width:3;opacity:.9}
@keyframes pd-run{0%{transform:translateX(0)}100%{transform:translateX(620px)}}
@keyframes pd-a{0%,6%{opacity:1}30%,100%{opacity:.12}}
@keyframes pd-b{0%,32%{opacity:.12}42%,64%{opacity:1}78%,100%{opacity:.12}}
@keyframes pd-c{0%,80%{opacity:.12}92%,100%{opacity:1}}
@keyframes pd-rope{0%,100%{opacity:.25}50%{opacity:.85}}
.pd-sweep{animation:pd-run 12s linear infinite}
.pd-l1{animation:pd-a 12s linear infinite}
.pd-l2{animation:pd-b 12s linear infinite}
.pd-l3{animation:pd-c 12s linear infinite}
.pd-rope{animation:pd-rope 3.2s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.pd-sweep,.pd-l1,.pd-l2,.pd-l3,.pd-rope{animation:none}.pd-l1,.pd-l2,.pd-l3{opacity:1}}
</style>
<rect class="pd-bg" x="70" y="30" width="620" height="250" rx="8"/>
<rect class="pd-prem" x="70" y="30" width="620" height="105"/>
<rect class="pd-disc" x="70" y="135" width="620" height="145"/>
<line class="pd-nav" x1="70" y1="135" x2="690" y2="135"/>
<text class="pd-s" x="118" y="128">NAV (0%)</text>
<polyline class="pd-line" points="70,92 130,74 190,60 250,86 310,128 350,150 410,196 470,238 520,248 560,214 620,168 660,138 690,136"/>
<line class="pd-ax" x1="70" y1="280" x2="690" y2="280"/>
<text class="pd-s" x="110" y="300">2015</text>
<text class="pd-s" x="250" y="300">2020</text>
<text class="pd-s" x="350" y="300">Feb 2021</text>
<text class="pd-s" x="500" y="300">Dec 2022</text>
<text class="pd-s" x="668" y="300">Jan 2024</text>
<text class="pd-t" x="190" y="50">peak premium +142%</text>
<text class="pd-t" x="512" y="268">−45% at 30 Dec 2022</text>
<rect class="pd-rope" x="70" y="30" width="590" height="250" fill="none" stroke="#f59f00" stroke-width="2" stroke-dasharray="10 8" rx="8"/>
<text class="pd-k pd-l1" x="200" y="345" fill="#2f9e44">PREMIUM — new shares locked 6 months, supply lags demand</text>
<text class="pd-k pd-l2" x="390" y="345" fill="#c92a2a">DISCOUNT — locked shares arrive, inflows stop, no redemption rope</text>
<text class="pd-k pd-l3" x="560" y="345" fill="#1c7ed6">PAR — ETF conversion restores redemption, gap closes</text>
</svg>
<figcaption>Schematic of the arc. The dated point is filed: a 45% discount as of 30 December 2022. The extremes over the whole closed-end era (5 May 2015 to 10 January 2024) were a +142% maximum premium and a −49% maximum discount — the filings report those as period extremes and do not date them, so the curve's shape between the labelled points is illustrative. The amber border is the missing redemption right, present for the entire premium era and the entire discount era, and removed only at the far right.</figcaption>
</figure>

### Worked example 3: what a 49% discount means per share

Percentages hide how brutal this is. Let us convert one into a share.

![At the record discount the bitcoin inside a share was unchanged while the wrapper's price was cut nearly in half.](/imgs/blogs/barry-silbert-dcg-and-the-genesis-contagion-5.webp)

Take a share whose NAV is a clean **\$100** of bitcoin.

Start with a **dated** reading rather than an extreme, because dated readings are what filings actually pin down. The FY2022 10-K states: "As of December 30, 2022, the last business day of the period, the Trust's Shares were quoted on OTCQX at a discount of **45%** to the Trust's Digital Asset Holdings per Share." At that level:

$$\text{Price} = \$100 \times (1 - 0.45) = \$55.00.$$

The worst reading the filings report for the whole closed-end era is a **49%** maximum discount — reported as a period extreme, without a date attached, so it is not attributable to any particular day. At that level:

$$\text{Price} = \$100 \times (1 - 0.49) = \$51.00.$$

You own \$100 of bitcoin. The market will give you \$51 for it. The gap is **\$49 per share** — value you demonstrably own and cannot reach, because there is no door. In an ETF this situation lasts minutes, because an arbitrageur buys your cheap share, redeems it for \$100 of bitcoin, and takes the \$49. In a closed-end trust with redemptions suspended, that trade does not exist, so the \$49 just sits there.

Now make it a balance-sheet problem. Suppose a fund holds \$500 million of GBTC and has borrowed against it:

- Marked at NAV: **\$500 million** of collateral.
- Marked at the market price at a 49% discount: **\$255 million**.
- Sold in size into the OTC market on a bad day: less again, because a large order walks down through the available bids.

Nothing happened to the bitcoin. The trust still held every coin. What collapsed was the *wrapper*, and the wrapper was the collateral.

The filings quantify how normal the discount became. Across the entire closed-end era — 5 May 2015 to 10 January 2024 — the shares closed at a discount on **725 separate days**, with an **average discount of 25%**.

**The one-sentence intuition:** in a closed-end fund the discount is not a temporary mispricing waiting to correct — with no redemption right, there is no force that makes it correct, so it can persist for years and deepen while you wait.

### The sponsor's parent started buying its own trust's shares

One more disclosure belongs here, because it shows how seriously the discount was taken inside the group. From the FY2023 10-K:

- On **10 March 2021**, the sponsor's board "approved the purchase by DCG, the parent company of the Sponsor, of up to \$250 million worth of Shares of the Trust."
- On **30 April 2021**, the board "approved the purchase by DCG of up to \$750 million worth" of shares.
- As of **19 February 2024**, "DCG and certain of its subsidiaries, including Genesis Global Capital, LLC, hold 7.0% of the Shares."

Read those dates against the arc. The premium flipped to a discount in **late February 2021** (CoinDesk). Within two weeks the parent authorized buying up to \$250 million of the shares; six weeks later, up to \$750 million more.

There is a benign reading and a less benign one, and honesty requires giving both. The benign reading: buying an asset trading below the value of its contents is a rational investment, and supporting a flagship product is a normal thing for a parent to do. The less benign reading: the parent buying into a falling discount is buying an illiquid, non-redeemable asset with a self-interest in how it is perceived. Nothing in the public record establishes which motive dominated, and no regulator has charged DCG over these purchases. What is established is the fact of them, disclosed in the trust's own filings.

## The lender: how Genesis converted a market into a balance sheet

Grayscale earned its fee no matter what. The part of DCG that could actually break was the lender.

### The business, and the size of its cushion

Genesis stood between people who had crypto and firms that wanted it. On the funding side its most retail-facing pipe was **Gemini Earn**, run with the Gemini exchange. On the lending side it lent to trading firms and funds.

Size the economics. Say a lender borrows **\$1,000,000,000** of crypto from savers at **5%** and lends it out at **9%**:

$$\$1{,}000{,}000{,}000 \times (0.09 - 0.05) = \$40{,}000{,}000 \text{ per year.}$$

Forty million dollars a year to stand in the middle. It looks wonderful — until you ask what happens if a borrower does not pay. If the lender holds \$100 million of equity against that \$1 billion book, the cushion is 10%. A single borrower larger than 10% of the book can therefore wipe out the owners entirely and start eating the savers' money.

That is not a hypothetical about Genesis. The New York Attorney General's October 2023 complaint **alleges** that "at one point, Sam Bankman-Fried's Alameda was the borrower for nearly 60 percent of all outstanding loans from Genesis to third parties." Sixty percent of a loan book in one counterparty is not lending; it is a single concentrated bet wearing a lender's costume. That allegation has not been adjudicated against DCG or Silbert.

### The retail funnel, and where the yield came from

The Gemini Earn mechanics are laid out in the SEC's January 2023 complaint. In **December 2020**, Genesis "entered into an agreement with Gemini to offer Gemini customers, including retail investors in the United States, an opportunity to loan their crypto assets to Genesis in exchange for Genesis' promise to pay interest." From **February 2021** the program went live, "with Gemini acting as the agent to facilitate the transaction." And: "Gemini deducted an agent fee, sometimes as high as **4.29 percent**, from the returns Genesis paid to Gemini Earn investors."

That 4.29% is the number that makes the whole product legible.

#### Worked example 4: a Gemini Earn depositor's yield, decomposed

![The depositor's yield was the last slice of a leveraged loan, after the agent fee and the lender's spread came out.](/imgs/blogs/barry-silbert-dcg-and-the-genesis-contagion-8.webp)

You deposit crypto into Gemini Earn and are advertised a rate of around **8%** (the NYAG describes Earn rates advertised up to roughly that level). Where does 8% come from? Work backwards through the stack. The intermediate splits below are illustrative — only the agent fee is sourced — but the *structure* is exactly as described in the SEC complaint.

1. **You lend to Genesis.** Not to "Gemini Earn", not to a bank, not to a fund. Genesis is the borrower, and Genesis is one company.
2. **Gemini takes an agent fee**, sometimes as high as **4.29%**, deducted from the returns Genesis paid (SEC complaint, 12 January 2023).
3. **Genesis keeps a lending spread** — its own compensation for standing in the middle. Call it 2 points for illustration.
4. **Therefore the ultimate borrower is paying** roughly:

$$8\% + 4.29\% + 2\% \approx 14.3\%.$$

Now ask the only question that matters: **who borrows crypto at ~14% a year, and why?** Not a homeowner. Not a corporation building a factory. The answer is a leveraged trading firm that expects to make more than 14% on the borrowed coins — for instance, by feeding them into a certain trust and selling the shares at a 37% premium six months later.

So the depositor's 8% was not a savings rate. It was the *residual slice* of a leveraged directional trade, after two intermediaries were paid, and the depositor carried the borrower's default risk while receiving the smallest share of the reward.

Compare the risk-return shapes honestly:

| Position | Best case | Worst case |
| --- | --- | --- |
| The leveraged borrower | Unlimited upside on the trade | Loses their own equity, defaults on the loan |
| Genesis (the lender) | Collects the spread | Loses the loan; its equity absorbs it first |
| Gemini (the agent) | Collects up to 4.29% | Reputational and legal exposure |
| **The Earn depositor** | **Receives ~8%** | **Loses the entire principal** |

The depositor has the worst payoff profile in the table: a capped, modest upside against a total loss. That is the shape of a *junior lender*, not a saver.

**The one-sentence intuition:** if you cannot name the ultimate borrower and explain why they can afford your yield, you are not earning interest — you are underwriting a trade you have not seen.

### The rating that changed and the marketing that did not

The NYAG's October 2023 complaint makes a specific, dated allegation about what Gemini knew. Per the Attorney General's office: "Only a year into the program, in February 2022, Gemini revised its estimate of Genesis' credit rating from BBB (investment grade) to CCC (junk grade) but did not publicly reveal to investors that it downgraded its rating and continued to market Earn as low-risk." The complaint further alleges that in July 2022 "Gemini's board of managers discussed ending the Gemini Earn program because of the risks associated with Genesis, and one board member even compared Genesis' financial condition to that of Lehman Brothers before its collapse."

Those are allegations from a civil complaint. Gemini settled with the Attorney General in June 2024 without the case going to judgment; the settlement terms are described below. But the structural point stands independent of intent: a retail depositor could not see the agent's internal credit rating of the borrower, and the marketing they *could* see said "low risk."

## June 2022: the loss that walked up the org chart

Now the two halves of the story meet.

### The default

By mid-2022 the GBTC premium had been gone for a year and the market was falling hard. Three Arrows Capital — a hedge fund that had built an enormous position around the premium trade, among many other levered bets — could not meet a margin call.

The SEC's own order, announced 17 January 2025, states the sequence: "in mid-June 2022, Three Arrows Capital, a crypto asset hedge fund and one of Genesis's largest borrowers, defaulted on a margin call, which compromised Genesis's business," and refers to "the approximately **\$1 billion loss**."

Reporting based on court filings put Genesis' gross exposure to 3AC at about **\$2.36 billion** (The Block, July 2022), against collateral that Genesis liquidated, and Genesis subsequently filed a claim of roughly **\$1.2 billion** against 3AC's estate (CoinDesk, 18 July 2022). 3AC entered liquidation proceedings in mid-2022. The full anatomy of that fund's leverage is its own story, told in [Su Zhu, 3AC, and the leverage that broke the lenders](/blog/trading/crypto-players/su-zhu-3ac-and-the-leverage-that-broke-the-lenders) and in [Three Arrows Capital and crypto-lender contagion](/blog/trading/crypto/three-arrows-capital-and-crypto-lender-contagion). Here we care only about what it did to Genesis.

### The patch

A roughly billion-dollar hole at a lender that owes money to retail savers on demand is an existential event. What happened next is the reason this post exists.

<figure class="blog-anim">
<svg viewBox="0 0 760 420" role="img" aria-label="A loss travels up an organization chart: Three Arrows Capital defaults, a red loss marker moves up into Genesis, then a promissory note marker moves down from Digital Currency Group into Genesis while the loss marker continues upward to the parent, illustrating that the hole was relocated rather than filled with cash" style="width:100%;height:auto;max-width:820px">
<style>
.oc-box{stroke:var(--border,#adb5bd);stroke-width:2;rx:8}
.oc-dcg{fill:#a5d8ff}
.oc-gen{fill:#d0bfff}
.oc-3ac{fill:#ffc9c9}
.oc-ret{fill:#e9ecef}
.oc-t{font:700 15px ui-sans-serif,system-ui;fill:#1f2937;text-anchor:middle}
.oc-sub{font:500 12.5px ui-sans-serif,system-ui;fill:#343a40;text-anchor:middle}
.oc-ax{stroke:var(--text-secondary,#868e96);stroke-width:2;fill:none}
.oc-cap{font:600 12.5px ui-sans-serif,system-ui;text-anchor:middle}
@keyframes oc-up1{0%,8%{transform:translateY(0);opacity:0}14%{opacity:1}34%,100%{transform:translateY(-104px);opacity:1}}
@keyframes oc-down{0%,40%{transform:translateY(0);opacity:0}46%{opacity:1}64%,100%{transform:translateY(104px);opacity:1}}
@keyframes oc-up2{0%,66%{transform:translateY(0);opacity:0}72%{opacity:1}88%,100%{transform:translateY(-104px);opacity:1}}
@keyframes oc-flash{0%,86%{opacity:.15}94%,100%{opacity:1}}
.oc-m1{animation:oc-up1 11s ease-in-out infinite}
.oc-m2{animation:oc-down 11s ease-in-out infinite}
.oc-m3{animation:oc-up2 11s ease-in-out infinite}
.oc-final{animation:oc-flash 11s linear infinite}
@media (prefers-reduced-motion:reduce){.oc-m1,.oc-m2,.oc-m3,.oc-final{animation:none;opacity:1;transform:none}}
</style>
<rect class="oc-box oc-dcg" x="250" y="26" width="260" height="66" rx="8"/>
<text class="oc-t" x="380" y="52">Digital Currency Group</text>
<text class="oc-sub" x="380" y="74">parent holding company</text>
<rect class="oc-box oc-gen" x="250" y="170" width="260" height="66" rx="8"/>
<text class="oc-t" x="380" y="196">Genesis Global Capital</text>
<text class="oc-sub" x="380" y="218">the lender — owes retail on demand</text>
<rect class="oc-box oc-3ac" x="250" y="314" width="260" height="66" rx="8"/>
<text class="oc-t" x="380" y="340">Three Arrows Capital</text>
<text class="oc-sub" x="380" y="362">borrower — defaults, mid-June 2022</text>
<rect class="oc-box oc-ret" x="30" y="170" width="180" height="66" rx="8"/>
<text class="oc-sub" x="120" y="196">Gemini Earn depositors</text>
<text class="oc-sub" x="120" y="216">SEC: ~$900M, 340,000</text>
<path class="oc-ax" d="M380 314 L380 236"/>
<path class="oc-ax" d="M380 170 L380 92"/>
<path class="oc-ax" d="M250 203 L210 203"/>
<g class="oc-m1"><rect x="392" y="270" width="150" height="30" rx="15" fill="#ffc9c9" stroke="#c92a2a" stroke-width="2"/><text class="oc-cap" x="467" y="290" fill="#c92a2a">~$1B loss</text></g>
<g class="oc-m2"><rect x="392" y="112" width="196" height="30" rx="15" fill="#d0bfff" stroke="#7048e8" stroke-width="2"/><text class="oc-cap" x="490" y="132" fill="#5f3dc4">$1.1B note — due 2032</text></g>
<g class="oc-m3"><rect x="150" y="126" width="150" height="30" rx="15" fill="#ffc9c9" stroke="#c92a2a" stroke-width="2"/><text class="oc-cap" x="225" y="146" fill="#c92a2a">~$1B loss</text></g>
<text class="oc-cap oc-final" x="380" y="406" fill="#c92a2a">Capital actually transferred to Genesis: $0 (SEC order, 17 Jan 2025)</text>
</svg>
<figcaption>The loss did not vanish — it moved. A defaulted borrower's hole travelled up into the lender, a ten-year IOU travelled down from the parent, and the hole continued upward to sit with DCG. The balance sheet balanced; no cash changed hands.</figcaption>
</figure>

What DCG did was issue a **promissory note** — a written promise to pay a fixed sum on a fixed date — to Genesis. The two authorities describe it consistently:

- The **SEC's order** (17 January 2025) refers to "DCG and Genesis enter[ing] into a promissory note with a **10-year term**."
- The **New York Attorney General** (amended complaint, 9 February 2024) is more specific: "after losing more than \$1.1 billion on loan defaults, Genesis, DCG, and their executives tried to conceal their losses by entering into a **\$1.1 billion promissory note**, in which DCG agreed to pay Genesis \$1.1 billion in a decade at only a **one percent** interest rate."

Per the amended complaint the note was **executed on 30 June 2022**, and the Attorney General alleges that **no principal or interest was ever paid on it**.

Separate the layers carefully. The *terms* — \$1.1 billion, ten years, 1% — are what the Attorney General's office states, and the ten-year term is independently confirmed in the SEC's settled order. The characterization "tried to conceal" is an **allegation in a civil complaint**, unproven and contested; it has not been tested at trial and DCG has not settled the Attorney General's claims. The allegation that nothing was ever paid on the note is likewise an allegation, though it sits comfortably alongside the SEC's own settled finding that no capital was transferred.

Then comes the sentence that defines the whole episode. From the SEC's press release describing its order: Moro, "with the knowledge and participation of DCG personnel—misleadingly tweeted that DCG had ensured that Genesis had 'adequate capital to operate' when DCG had in fact **not transferred any capital to Genesis**."

#### Worked example 5: what a 10-year, 1% note is actually worth

A promissory note has a face value. That is not its value. Value depends on *when* you get the money and *how likely* you are to get it.

![The note said $1.1 billion; discounting a 1% coupon over ten years at any realistic rate leaves less than half that value today.](/imgs/blogs/barry-silbert-dcg-and-the-genesis-contagion-7.webp)

The tool is **present value**: a dollar arriving in ten years is worth less than a dollar today, because today's dollar could have been earning a return in the meantime. For a note paying an annual coupon $C$ for $n$ years and returning face value $F$ at the end, discounted at rate $r$:

$$PV = \sum_{t=1}^{n} \frac{C}{(1+r)^t} + \frac{F}{(1+r)^n}$$

where $C$ is the annual interest payment, $F$ is the face value repaid at maturity, $r$ is the discount rate demanded for that risk, and $n$ is the number of years.

Our note: $F = \$1.1$ billion, $n = 10$, and a 1% coupon means $C = \$11$ million a year.

**At a 10% discount rate.** The coupon stream is an annuity of \$11 million for 10 years:

$$\$11\text{m} \times \frac{1 - (1.10)^{-10}}{0.10} = \$11\text{m} \times 6.1446 = \$67.6\text{m}.$$

The principal:

$$\frac{\$1{,}100\text{m}}{(1.10)^{10}} = \frac{\$1{,}100\text{m}}{2.5937} = \$424.1\text{m}.$$

Total: **\$491.7 million** — about **45%** of face.

**At a 15% discount rate** (arguably more appropriate for an unsecured claim on a crypto holding company in mid-2022):

$$\$11\text{m} \times 5.0188 = \$55.2\text{m}, \qquad \frac{\$1{,}100\text{m}}{4.0456} = \$271.9\text{m}.$$

Total: **\$327.1 million** — under **30%** of face.

So the hole was about \$1.1 billion, and the thing placed into the hole was worth roughly **\$327–492 million** on a present-value basis, before even asking whether DCG could pay in 2032. The gap between the face value and the economic value is the part that never got filled.

And notice what changed on Genesis' balance sheet. Before: a loan to an independent third party. After: a claim **on its own parent**. The asset moved from an arm's-length exposure to a related-party exposure — the exact category regulators scrutinize hardest, and the one an outside depositor had no way to evaluate.

**The one-sentence intuition:** face value is a number printed on paper; value is that number adjusted for when it arrives and how likely it is to arrive, and for a ten-year note at 1% those adjustments take away more than half.

## November 2022 to January 2023: the run, the halt, the filing

Balance sheets fail slowly and then liquidity fails all at once.

### The run

In November 2022 FTX collapsed, and every crypto lender faced the question depositors ask simultaneously: *is my money there?* Genesis disclosed that it had funds locked in its FTX trading account — around **\$175 million**, per its own statement of 10 November 2022 as widely reported — and DCG was reported to have made an equity infusion of about **\$140 million** into Genesis.

Then the withdrawals came. On **16 November 2022**, Genesis suspended withdrawals from its lending business, which froze the Gemini Earn program. The SEC's complaint describes it: "in November 2022, Genesis announced that it would not allow its Gemini Earn investors to withdraw their crypto assets because Genesis lacked sufficient liquid assets to meet withdrawal requests following volatility in the crypto asset market. At the time, Genesis held approximately **\$900 million** in investor assets from **340,000** Gemini Earn investors."

#### A source conflict worth naming

You will encounter two different sets of Earn numbers in the wild, and they do not obviously reconcile. Both are primary, and neither is wrong.

| Source | Investors | Amount |
| --- | --- | --- |
| SEC complaint, 12 January 2023 | **340,000** Earn investors | **~\$900 million** held by Genesis |
| NYAG, October 2023 onward | **more than 230,000** investors (incl. 29,000+ New Yorkers) | **more than \$1 billion**; the May 2024 settlement release refers to more than **\$1.1 billion** contributed through Earn |

They measure different things. The SEC figure is a **snapshot at the moment of the halt** — how many accounts held how much on that day. The Attorney General's figure is a **claims-based tally** built afterwards from investors who came forward, denominated in what they put in rather than what was frozen, and confined to the population the AG's office was acting for. A count of open accounts and a count of substantiated claimants are simply not the same number, and an amount frozen on one day is not the same as an amount contributed over two years.

If you have seen both numbers cited and wondered which is "the real one" — neither is a correction of the other. Whenever two regulators publish different totals for the same event, the first question is what each one was counting.

### Worked example 6: why the balance sheet balanced and the money still was not there

This is the crux, and it is a *calendar* problem rather than an arithmetic one.

![Genesis owed cash on demand while its two largest claims were a defaulted borrower and a ten-year IOU from its own parent.](/imgs/blogs/barry-silbert-dcg-and-the-genesis-contagion-6.webp)

Lay out the simplified book at the moment of the freeze.

**What Genesis was owed (assets):**

| Asset | Amount | When payable | Quality |
| --- | --- | --- | --- |
| Claim against 3AC | ~\$1.2B filed | Through a bankruptcy estate, years | Defaulted |
| Promissory note from DCG | \$1.1B at 1% | **2032** | Related party |
| Other loans | Various | Various | Mixed |

**What Genesis owed (liabilities):**

| Liability | Amount | When payable |
| --- | --- | --- |
| Gemini Earn investors | ~\$900M | **On demand** |
| Institutional creditors | Various | Mostly short-dated |

Add the assets and the total may well exceed the liabilities. On paper, solvent. Now ask the depositor's question: *a saver wants \$5,000 back this week — which asset pays them?*

- The 3AC claim? It is in a liquidation proceeding and will pay cents on the dollar, years from now.
- The DCG note? It matures in 2032 and pays 1% a year in the meantime.
- The other loans? To counterparties in the same collapsing market.

None of them produce \$5,000 this week. This is a **maturity mismatch** — assets that pay late funding liabilities that are due now — compounded by a **quality mismatch**, because the two largest assets were impaired or related-party. Every bank in history that has failed has failed this way; what made this case distinctive is that one of the two big assets was a claim on the bank's own owner.

**The one-sentence intuition:** solvency is about whether the numbers add up, liquidity is about whether the money is there on the day it is asked for, and depositors are only ever paid out of the second one.

### The filings

The sequence from here is public record:

- **12 January 2023** — the SEC charges Genesis Global Capital and Gemini Trust Company in the Southern District of New York with the unregistered offer and sale of securities through Gemini Earn, under Sections 5(a) and 5(c) of the Securities Act of 1933. The complaint notes that Gemini terminated the Earn program earlier that month.
- **19 January 2023** — Genesis Global Holdco, LLC and certain subsidiaries file voluntary petitions for Chapter 11 reorganization in the US Bankruptcy Court for the Southern District of New York, **Case No. 23-10063** (petition date per the court-appointed administrator's docket).
- The bankruptcy generates a long tail of adversary proceedings between the estate and its own parent, including *Genesis Global Capital v. Digital Currency Group* (23-01168) and *Genesis Global Capital v. DCG International Investments* (23-01169), with further actions between the estate and DCG filed as late as 2025 and 2026 on the same docket.

Alongside the filings ran a public argument. On 2 January 2023 Cameron Winklevoss published an open letter to Barry Silbert on behalf of Earn creditors; Silbert responded publicly the same day; a second, sharper letter followed on 10 January 2023 calling for Silbert's removal as CEO. Each side accused the other of misrepresenting the state of Genesis. That exchange is *contested characterization by opposing parties in a dispute*, and it is worth reading as such rather than as evidence — both firms were, at the time, negotiating over who would bear a loss.

## The reckoning: one set of facts, four proceedings

The same events produced four separate legal theories in four separate forums. Keeping them distinct matters, because they establish very different things.

![Four proceedings on four theories from one set of facts — and three of the four resolved without any admission of wrongdoing.](/imgs/blogs/barry-silbert-dcg-and-the-genesis-contagion-9.webp)

**1. SEC v. Genesis and Gemini — unregistered securities.** Filed 12 January 2023. The theory is narrow and technical: Earn was an offer and sale of securities that should have been registered, so investors were denied the disclosures registration would have forced. On **19 March 2024** the SEC announced that Genesis "agreed to a final judgment ordering it to pay a **\$21 million** civil penalty and imposing a permanent injunction." Notably, "the SEC will not receive any portion of the penalty until after payment of all other allowed claims by the bankruptcy court, including claims by retail investors in the Gemini Earn program" — the regulator explicitly put itself behind the depositors.

**2. SEC v. DCG and Moro — misleading statements.** Announced **17 January 2025**. DCG and Soichiro "Michael" Moro agreed to pay **\$38.5 million** combined — **\$38 million** and **\$500,000** respectively — to settle charges of misleading investors about Genesis' financial condition. The legal detail matters enormously: the settled charge is **Section 17(a)(3) of the Securities Act of 1933**, and both parties settled **"without admitting or denying the SEC's findings."**

Section 17(a)(3) is a **negligence-based** provision. It does not require proving intent to defraud, which the fraud provisions requiring *scienter* do. That is a meaningful distinction and it cuts in DCG's favor: the SEC resolved this as a negligence matter, not as intentional fraud, and DCG admitted nothing.

**3. NYAG v. Gemini, Genesis, DCG, Silbert and Moro — state fraud.** Filed **19 October 2023** alleging that the defendants defrauded "more than 230,000 investors, including at least 29,000 New Yorkers, of more than \$1 billion." Expanded by amended complaint on **9 February 2024** to add "an additional \$2 billion," bringing the total alleged to "more than 230,000 investors out of more than \$3 billion." Resolutions:

- **20 May 2024** — a settlement "worth \$2 billion" with the Genesis entities, creating a "Victims' Fund" that "will receive distributions from the assets remaining in Genesis' estate after initial bankruptcy distributions to creditors," and banning Genesis from operating in New York. Critically: "Under this settlement, Genesis **neither admits nor denies** the allegations of this lawsuit, and the suit will continue against the remaining defendants."
- **14 June 2024** — approximately **\$50 million** recovered from Gemini, which the AG's office describes as providing "all defrauded investors full recovery of the assets they invested in the Earn program." Gemini was banned from operating any crypto lending program in New York and required to cooperate with the OAG's continuing litigation "against Digital Currency Group (DCG), DCG's CEO Barry Silbert, and Genesis' former CEO Soichiro Moro."

That last clause is important for fairness: as of those announcements, the Attorney General's claims **against DCG, Silbert and Moro remained unresolved**. Allegations in a complaint are not findings, and this post does not treat them as such.

**4. In re Genesis Global Holdco — Chapter 11.** Filed **19 January 2023**. The plan's **effective date was 2 August 2024** per the administrator's docket. Because the estate's recoveries were substantially in kind — crypto returned as crypto — and because bitcoin and ether rose sharply between the January 2023 filing and the 2024 distributions, Earn creditors' dollar-denominated outcomes were far better than the freeze-day marks implied. This is a genuinely unusual feature of the case and it deserves emphasis: **most depositors in this story were eventually made whole in coin terms**, which is not what "crypto lender bankruptcy" usually means. It also does not retroactively make the structure safe. Recovering because the collateral asset tripled is luck, not risk management.

## How it shows up in price

Everything above eventually lands on a screen as a number. Here is the transmission.

### The discount as a stress gauge

Once GBTC was in a persistent discount, the discount itself became a market signal — and a market force.

It signalled **who was trapped**. A wide discount is direct evidence that large holders cannot exit at NAV and that nobody wants the wrapper. Any fund marking GBTC at NAV looked solvent; the same fund marking it at the traded price looked wounded. Because 3AC and others had borrowed against GBTC, the discount degraded the value of collateral across the whole lending complex simultaneously.

It was **reflexive**. Forced sellers sold into a thin OTC market, which widened the discount, which impaired collateral further, which forced more selling. Reflexivity is the property that makes a price move self-reinforcing rather than self-correcting, and a closed-end discount with no redemption valve is close to a pure example.

And it was **a bet on a regulator**. Because the discount could only close if the trust became redeemable, holding GBTC at a discount was implicitly a wager on a regulatory outcome, with an unknown date. That is a very different exposure from "long bitcoin," and many holders did not distinguish the two.

### Forced selling and the slippage tax

When a borrower defaults, the lender liquidates collateral, and liquidation is the worst kind of selling: maximum size, maximum urgency, no discretion about timing.

Picture an order book. Suppose buyers stand for 100,000 tokens at \$20, another 100,000 at \$19, another 100,000 at \$18, and so on. If you must sell 500,000 tokens now, you do not get \$20 for all of them — you walk down through each level and your *average* fill might be \$18. That 10% shortfall is the **slippage tax**, and it is paid straight into the market price for everyone else. Genesis liquidating a defaulted borrower's collateral in mid-2022 was exactly this: one fund's leverage becoming every holder's lower price.

### The convergence trade

Then the catalyst arrived, and the arithmetic that had punished holders for three years paid the people who bought into it.

The path is documented in the trust's own 10-K:

- **June 2022** — the SEC issues a final order disapproving NYSE Arca's rule change to list GBTC as an exchange-traded product. The sponsor petitions for review in the DC Circuit.
- **August 2023** — "the D.C. Circuit Court of Appeals granted the Sponsor's petition and **vacated the SEC's order as arbitrary and capricious**. The SEC did not seek panel rehearing or rehearing en banc."
- **October 2023** — the court remands the matter to the SEC.
- **10 January 2024** — "the SEC approved an application under Rule 19b-4 ... by NYSE Arca to list the Shares of the Trust."
- **11 January 2024** — "Shares of the Trust began trading on NYSE Arca under the symbol 'GBTC'." The sponsor commenced the redemption program under Regulation M exemptive relief, and the sponsor's fee dropped: "From inception to January 10, 2024, the Sponsor's Fee was 2.0%. Effective January 11, 2024, the Sponsor's Fee was lowered to **1.5%**."

#### Worked example 7: the convergence trade

The best part of this example is that we do not have to invent the entry price. The filings give a dated series of discount readings, each on the last business day of its period:

| Date | Discount to NAV | Venue |
| --- | --- | --- |
| 30 December 2022 | **45%** | OTCQX |
| 29 December 2023 | **8%** | OTCQX |
| 19 February 2024 | **0.02%** | NYSE Arca |
| 31 December 2025 | **0.07%** | NYSE Arca |

Buy at the first row, hold to the third. Bitcoin's own move is a separate matter; isolate just the *wrapper*.

You buy **\$100** of NAV for:

$$\$100 \times (1 - 0.45) = \$55.$$

At conversion the discount effectively vanishes, so your share is worth its NAV. Ignoring the fee for a moment, the wrapper alone returns:

$$\frac{\$100 - \$55}{\$55} = 0.818 = 81.8\%.$$

Now net out the friction. Holding from 30 December 2022 to the conversion is just over twelve months, and at a 2% annual fee NAV erodes about 2.1%, so the NAV you actually receive is nearer \$97.90:

$$\frac{\$97.90 - \$55}{\$55} = 0.780 = 78.0\%.$$

Roughly a **78% return from the structure alone**, on top of whatever bitcoin did — the mirror image of the premium trade that started the story, and available for exactly the same reason: a gap between price and NAV that a mechanism eventually forced shut.

And the convergence was not approximate. Over the whole NYSE Arca era from **11 January 2024 to 31 December 2025**, the filings report a **maximum premium of 1.68%** (average 0.06%) and a **maximum discount of 1.56%** (average 0.08%). A wrapper that had spent nine years wandering between +142% and −49% now sits inside a band of under two points in either direction. That is what a redemption right does, stated as precisely as it can be stated.

The catch, and it is the entire risk: **the date was unknowable**. A buyer at a 35% discount in mid-2022 watched it deepen to 45% by the end of December before it closed, and the closing required a favorable appellate ruling that had not yet happened. Sizing a position whose payoff depends on a court is a different exercise from sizing a directional bet, and a discount can always widen further while you are right about the destination.

### And then the flows reversed

Conversion also created selling. Holders trapped for three years could finally exit at NAV, and many did. Because redemptions in a spot bitcoin ETF are settled by selling the underlying, GBTC outflows translated into actual bitcoin hitting the market in early 2024, at the same time as the newly launched competing ETFs were buying. A structural feature of one wrapper became a real bid and offer in the whole asset class — the bridge traced in [Bitcoin ETFs and the TradFi bridge](/blog/trading/crypto/bitcoin-etfs-and-the-tradfi-bridge).

There is an irony worth sitting with. The fee cut from 2.0% to 1.5% arrived at the exact moment GBTC finally faced competitors — several of which launched at a fraction of that. A 2% fee was defensible when the product was the only door; it was not once the door was one of eleven.

## Common misconceptions

**"A discount means the fund is cheap — just buy it below NAV."** Only if something will *force* convergence. In an open-end ETF, redemption forces it daily. In a closed-end trust, nothing does, so a discount can persist for years and deepen while you hold. GBTC traded at a discount on 725 separate days between 5 May 2015 and 10 January 2024, with an average discount of 25%. The people who eventually profited were right about the mechanism *and* patient enough to survive an unknown wait — and they needed an appellate ruling to get paid.

**"Grayscale blew up."** It did not. GBTC held its bitcoin throughout, published its holdings, filed its 10-Ks, and Grayscale collected its 2% the entire time. The trust was arguably the safest thing in the group. What broke was **Genesis**, the lender. GBTC's role was as the *engine* of the leveraged trade and as *collateral* in the lending book — so its discount hurt the borrowers, not the trust.

**"The \$1.1 billion note means DCG paid Genesis \$1.1 billion."** No. A promissory note is a promise to pay later. The SEC's order states directly that DCG "had in fact not transferred any capital to Genesis" at the time it was described as having ensured Genesis had adequate capital to operate. The note changed what Genesis' balance sheet *said* without changing what Genesis *had*.

**"DCG was found guilty of fraud."** It was not, and the distinction is not a technicality. DCG settled with the SEC under Section 17(a)(3), a **negligence-based** provision that does not require intent to deceive, and did so **without admitting or denying** the findings. The New York Attorney General's fraud allegations against DCG and Barry Silbert were, as of the AG's own June 2024 announcement, still being litigated rather than resolved. Allegations in a complaint are not findings of fact.

**"Barry Silbert was charged over this."** He was not. The Attorney General's action is a **civil** suit; a civil complaint alleging fraud is not a criminal charge, is decided on a lower standard of proof, and carries no possibility of imprisonment. No criminal charges against Silbert are on the public record. The vocabulary matters: "sued", "charged", and "convicted" describe three completely different things, and only the first applies here.

**"The Earn depositors were speculating and knew the risks."** Many understood Earn as a high-yield savings product; the SEC's charge was precisely that the registration and disclosure regime that would have told them otherwise was bypassed. And the specific facts the NYAG alleges were invisible to them — an internal credit downgrade from BBB to CCC, a loan book at one point ~60% concentrated in a single counterparty — are not things a retail user could have discovered from the app.

**"A conglomerate is safer because it's diversified."** Diversification helps only when the parts are *uncorrelated*. DCG's arms were levered to the same asset, the same cycle, and often literally the same collateral. Correlated diversification is concentration wearing a costume — with extra pipes for a loss to travel through.

**"The depositors lost everything."** In dollar terms, most did not, and honesty requires saying so. Genesis' plan went effective on 2 August 2024, recoveries were substantially in kind, and crypto prices rose sharply between the freeze and the distributions. But being made whole because the asset you were owed happened to triple is an outcome, not a safeguard. Had bitcoin fallen over those twenty months, the same structure would have produced a catastrophe.

## How it shows up in real markets

### The case study: Gemini Earn, from launch to settlement

The Earn sequence is worth assembling in one place, because it is the cleanest example on record of related-party plumbing converting one counterparty loss into a retail solvency crisis.

**December 2020** — Genesis and Gemini sign the agreement (SEC complaint). **February 2021** — Earn launches; Gemini acts as agent and deducts a fee "sometimes as high as 4.29 percent" from returns. **Late February 2021** — GBTC's premium flips to a discount (CoinDesk), quietly removing the trade that made high crypto borrowing rates economic. **10 March and 30 April 2021** — the sponsor's board approves DCG purchases of GBTC of up to \$250 million and then up to \$750 million (10-K). **February 2022** — Gemini internally revises its estimate of Genesis' credit rating from BBB to CCC and, per the NYAG, continues marketing Earn as low-risk. **May–June 2022** — the market falls; **mid-June 2022**, 3AC defaults on a margin call (SEC order), producing "approximately \$1 billion" of loss. **30 June 2022** — DCG and Genesis execute the promissory note: \$1.1 billion, ten years, 1%, with no principal or interest ever paid on it (NYAG amended complaint); no capital transferred to Genesis (SEC order). **July 2022** — Gemini's board discusses ending Earn; one member compares Genesis to Lehman (NYAG allegation). **November 2022** — FTX collapses; Genesis discloses ~\$175 million locked at FTX (reported); **16 November 2022**, withdrawals halt, freezing ~\$900 million belonging to 340,000 Earn investors (SEC). **12 January 2023** — SEC charges Genesis and Gemini. **19 January 2023** — Genesis files Chapter 11 (Case 23-10063). **19 October 2023** — NYAG sues. **August 2023 / 10 January 2024** — the DC Circuit vacates the SEC's ETF denial; the SEC approves the listing; **11 January 2024** GBTC begins trading on NYSE Arca, the redemption program restarts, the fee drops to 1.5%, and the discount closes (0.02% by 19 February 2024). **9 February 2024** — NYAG amends to \$3 billion. **19 March 2024** — Genesis settles with the SEC for \$21 million, subordinated to retail claims. **20 May 2024** — \$2 billion NYAG settlement with Genesis, no admission. **14 June 2024** — ~\$50 million from Gemini, no admission, litigation continues against DCG, Silbert and Moro. **2 August 2024** — the Chapter 11 plan goes effective. **17 January 2025** — DCG and Moro settle with the SEC for \$38.5 million combined, without admitting or denying.

Three and a half years, four forums, and the mechanism at the center is a single sentence: **a loss at an arm's-length borrower was moved onto a related party's promise, and the people funding the lender could not see it.**

### The same shape elsewhere

**Closed-end fund discounts are ordinary in traditional markets.** Listed closed-end funds routinely trade at 5–15% discounts to NAV for years — the phenomenon is old enough to have a literature and a name ("the closed-end fund puzzle"). What made GBTC extreme was the combination of a volatile underlying, a six-month lockup, a ratcheting share count, and a single-asset moat that eroded. The mechanism was not exotic; the magnitude was.

**Conglomerate finance arms have done this before.** The most instructive precedent is a finance subsidiary funding itself on the strength of its industrial parent's name. When the parent's implicit support turns out to be worth less than the market assumed, funding evaporates faster than assets can be sold. The lesson repeated in 2022 is that *implicit* support is not a contract; it is a reputation, and reputations reprice instantly.

**Bank regulation exists largely to prevent this exact pattern.** In US banking, transactions between a bank and its affiliates are constrained by law: there are quantitative caps on such exposures and collateral requirements for extensions of credit to affiliates. The reason those rules exist is precisely that a bank funded by depositors should not be able to convert their money into a claim on its own parent. Genesis was not a bank and was not subject to those rules, so it could hold a related-party note as a principal asset while owing retail on demand. That is not a loophole anyone exploited — it is a description of what an unregulated lender is.

**And crypto specifically keeps rebuilding it.** Every cycle produces new firms in which the exchange, the lender, the market maker, and the token issuer share an owner. The [FTX collapse](/blog/trading/crypto/ftx-collapse-sam-bankman-fried) is the extreme version. The general map of who profits at each link is in [Cui bono: the incentive map of crypto](/blog/trading/crypto-players/cui-bono-the-incentive-map-of-crypto), and the comparative view of DCG against the pure-fund model is in [Pantera and DCG: the crypto conglomerates](/blog/trading/crypto-players/pantera-dcg-and-the-crypto-conglomerates).

## What an outside depositor could actually have seen

The fair question is not "should someone have predicted the default?" — nobody could. It is narrower and more useful: **what was visible from outside, without inside information?**

More than you would expect. Four of the five checks below were answerable from public filings.

![Four of the five warning signs were answerable from documents anyone could read.](/imgs/blogs/barry-silbert-dcg-and-the-genesis-contagion-10.webp)

**1. Can I get out at NAV?** For GBTC, no — the 10-K stated plainly that redemptions were suspended in October 2014 and that "there is no arbitrage mechanism." A wrapper you cannot redeem is a wrapper whose price is a matter of opinion. Redeemability is the single most under-appreciated feature of any fund product.

**2. Who, specifically, is borrowing my money?** For Gemini Earn the answer was: one company, Genesis. Not a diversified book of loans, not a bank with deposit insurance — one counterparty. "Where does the yield come from?" has a specific answer or it has no answer, and the second case is itself information.

**3. Is my counterparty an affiliate of the product I'm buying?** GBTC's filings disclosed that the sponsor and the trust's sole authorized participant were both DCG subsidiaries, and later that DCG entities held 7.0% of the shares. Affiliation is not wrongdoing. It *is* a reason to discount every reassurance that comes from inside the group, because everyone speaking is on the same side of the trade.

**4. Does the yield have an arithmetic explanation?** Work the stack. If you receive 8% and the agent takes up to 4.29%, the borrower must be paying comfortably into the teens. Ask what business supports a mid-teens borrowing cost in a market with no obvious mid-teens risk-free return. The honest answer in 2021 — leveraged directional trading — was available to anyone who did the subtraction.

**5. Is concentration disclosed?** For Genesis it was not, and *that* was the signal. A lender that will not tell you its largest-exposure percentage is asking you to assume it is small. The NYAG later alleged one borrower reached nearly 60% of the third-party loan book. Absence of disclosure on the single most important risk metric is a data point, not a gap.

One more, which is a stance rather than a check: **a parent's promise is not capital.** When a group announces that it "stands behind" a subsidiary, ask whether cash moved. In this case the regulator's own finding was that it had not. "We have committed to support" and "we have transferred" are different sentences, and only one of them pays a withdrawal.

## When this matters to you

You will very likely never lend to a crypto prime broker. But the shapes here are not crypto shapes, and three of them will find you.

**Any wrapper you cannot redeem is priced by sentiment, not contents.** This applies to listed closed-end funds, some structured products, interval funds, and any tokenised claim whose redemption is discretionary. Before buying a fund, the first question is not what it holds — it is what happens when you want out, and whether anyone is contractually obliged to give you the underlying value. That question separates the ETF from the trust, and it was worth 49 points at the bottom.

**Yield is a description of risk, not a feature of a product.** Every above-market yield is a loan to someone at that rate. The chain always terminates in a borrower who must earn more than they pay. If you cannot name them, you have not bought a savings product — you have made an unsecured loan to a stranger and accepted the worst payoff in the chain.

**Ownership maps tell you where the walls are missing.** When one owner sits above the venue, the lender, the sponsor, and the outlet reporting on all three, the independent checks you are implicitly relying on may not exist. The map of that structure across crypto is in [crypto VCs and market makers](/blog/trading/crypto/crypto-vc-and-market-makers) and in [exchanges are players, not just venues](/blog/trading/crypto-players/exchanges-are-players-not-just-venues).

The final thing worth carrying is the smallest number in the post. Not \$2.36 billion, not \$1.1 billion, not 340,000 depositors. It is the SEC's finding that when Genesis was described as having been given adequate capital to operate, the capital actually transferred was **zero**. Everything else in this story — the trust, the fee, the lockup, the premium, the discount, the note, the freeze — is elaboration on the difference between a balance sheet that balances and money that is actually there.

*This post explains mechanisms and public record; it is educational material, not investment advice.*

## Sources & further reading

Primary sources first. Where a figure is from press reporting rather than a filing, the outlet and date are given inline above.

**Filings and court records**

- **Grayscale Bitcoin Trust, Form 10-K for FY2022** (filed 1 March 2023) — the 2.0% sponsor's fee "payable in Bitcoins"; the Rule 144 six-month holding period; the 28 October 2014 redemption suspension and its Regulation M origin; Genesis as sole authorized participant; the "no arbitrage mechanism" risk factor; premium/discount statistics to 31 December 2022; 0.1 bitcoin per share in the initial baskets; the 91-for-1 share split of 26 January 2018; 25 September 2013 first creation basket; and the dated reading that "as of December 30, 2022 ... the Trust's Shares were quoted on OTCQX at a discount of 45%", alongside 468 discount days to that point. [sec.gov](https://www.sec.gov/Archives/edgar/data/1588489/000119312523054302/d453116d10k.htm)
- **Grayscale Bitcoin Trust, Form 10-K for FY2023** (filed 23 February 2024) — the dated discount readings of 8% as of 29 December 2023 and 0.02% as of 19 February 2024; the DC Circuit vacating the SEC's order in August 2023 and the October 2023 remand; SEC approval of the NYSE Arca 19b-4 application on 10 January 2024 and the start of NYSE Arca trading on 11 January 2024; the fee reduction from 2.0% to 1.5% effective 11 January 2024; DCG board approvals of up to \$250 million (10 March 2021) and up to \$750 million (30 April 2021) of share purchases; DCG entities holding 7.0% of shares as of 19 February 2024; the "absence of arm's-length negotiation" risk factor; Barry Silbert's biography. [sec.gov](https://www.sec.gov/Archives/edgar/data/1588489/000119312524043678/d736998d10k.htm)
- **Grayscale Bitcoin Trust ETF, Form 10-K for FY2025** (filed 25 February 2026) — the complete closed-end era statistics (5 May 2015 to 10 January 2024: maximum premium 142%, average premium 37%, maximum discount 49%, average discount 25%, quoted at a discount on 725 days); the post-conversion statistics (11 January 2024 to 31 December 2025: maximum premium 1.68%, average 0.06%; maximum discount 1.56%, average 0.08%; 0.07% discount as of 31 December 2025); and the current 1.5% sponsor's fee, "generally paid in Bitcoin". Note that the filings report the 142% and 49% extremes as period maxima **without attaching dates to them**. [sec.gov](https://www.sec.gov/Archives/edgar/data/1588489/000119312526071956/gbtc-20251231.htm)
- **In re Genesis Global Holdco, LLC, et al.**, Case No. 23-10063, US Bankruptcy Court for the Southern District of New York — petition date 19 January 2023; plan effective date 2 August 2024; adversary proceedings including *Genesis Global Capital v. Digital Currency Group* (23-01168) and *Genesis Global Capital v. DCG International Investments* (23-01169). Docket via the court-appointed administrator. [restructuring.ra.kroll.com/genesis](https://restructuring.ra.kroll.com/genesis/)

**Regulatory actions**

- **SEC**, *SEC Charges Genesis and Gemini for the Unregistered Offer and Sale of Crypto Asset Securities through the Gemini Earn Lending Program*, press release 2023-7, 12 January 2023 — the December 2020 agreement; February 2021 launch; the Gemini agent fee "sometimes as high as 4.29 percent"; ~\$900 million from 340,000 Earn investors; Securities Act §§5(a) and 5(c). [sec.gov](https://www.sec.gov/news/press-release/2023-7)
- **SEC**, *Genesis Agrees to Pay \$21 Million Penalty to Settle SEC Charges*, press release 2024-37, 19 March 2024 — the penalty and its subordination to retail claims in the bankruptcy. [sec.gov](https://www.sec.gov/newsroom/press-releases/2024-37)
- **SEC**, *SEC Charges Digital Currency Group and Soichiro "Michael" Moro ... for Misleading Investors about Genesis's Financial Condition*, press release 2025-22, 17 January 2025 — the mid-June 2022 3AC default; the "approximately \$1 billion loss"; the 10-year promissory note; the finding that DCG "had in fact not transferred any capital to Genesis"; \$38 million and \$500,000 penalties under Securities Act §17(a)(3), settled without admitting or denying. [sec.gov](https://www.sec.gov/newsroom/press-releases/2025-22)
- **New York Attorney General**, *AG James Sues Cryptocurrency Companies Gemini, Genesis, and DCG for Defrauding Investors*, 19 October 2023 — 230,000+ investors, 29,000+ New Yorkers, \$1 billion+; the alleged Alameda concentration of "nearly 60 percent"; the alleged BBB-to-CCC internal downgrade in February 2022. [ag.ny.gov](https://ag.ny.gov/press-release/2023/attorney-general-james-sues-cryptocurrency-companies-gemini-genesis-and-dcg)
- **New York Attorney General**, *AG James Expands Lawsuit Against ... Digital Currency Group*, 9 February 2024 — the amended complaint; the \$1.1 billion promissory note "in a decade at only a one percent interest rate", executed 30 June 2022, on which the complaint alleges no principal or interest was ever paid; the \$3 billion total. These are allegations in a civil complaint that remains unresolved as against DCG and Silbert. [ag.ny.gov](https://ag.ny.gov/press-release/2024/attorney-general-james-expands-lawsuit-against-cryptocurrency-company-digital)
- **New York Attorney General**, *AG James Secures Settlement Worth \$2 Billion from Crypto Firm Genesis Global Capital*, 20 May 2024 — the Victims' Fund; the New York ban; "Genesis neither admits nor denies". [ag.ny.gov](https://ag.ny.gov/press-release/2024/attorney-general-james-secures-settlement-worth-2-billion-crypto-firm-genesis)
- **New York Attorney General**, *AG James Recovers \$50 Million from Crypto Firm Gemini for Defrauded Investors*, 14 June 2024 — the recovery, the lending ban, and the continuation of litigation against DCG, Silbert and Moro. [ag.ny.gov](https://ag.ny.gov/press-release/2024/attorney-general-james-recovers-50-million-crypto-firm-gemini-defrauded)

**Press reporting (for figures not in the filings above)**

- **The Block**, July 2022 — Genesis' reported ~\$2.36 billion exposure to Three Arrows Capital, based on court filings.
- **CoinDesk**, 18 July 2022 — Genesis' ~\$1.2 billion claim against the 3AC estate; also CoinDesk's contemporaneous coverage of the late-February 2021 premium-to-discount flip and the January 2024 close to par.
- **CNBC / CoinDesk**, 1 November 2021 — DCG's reported valuation above \$10 billion in its secondary share sale.
- **Contemporaneous reporting** (Reuters, Bloomberg, CoinDesk and Genesis' own statements), November 2022 — Genesis' ~\$175 million locked at FTX, DCG's reported ~\$140 million equity infusion, and the 16 November 2022 withdrawal halt. These are reported figures, not filed ones; treat them as attributed rather than audited.
- The Winklevoss–Silbert open letters of 2 January and 10 January 2023, published by both parties — read as contested statements by opposing sides of a dispute.

**Related posts on this site**

- [Su Zhu, 3AC, and the leverage that broke the lenders](/blog/trading/crypto-players/su-zhu-3ac-and-the-leverage-that-broke-the-lenders)
- [Three Arrows Capital and crypto-lender contagion](/blog/trading/crypto/three-arrows-capital-and-crypto-lender-contagion)
- [Bitcoin ETFs and the TradFi bridge](/blog/trading/crypto/bitcoin-etfs-and-the-tradfi-bridge)
- [Pantera and DCG: the crypto conglomerates](/blog/trading/crypto-players/pantera-dcg-and-the-crypto-conglomerates)
- [Cui bono: the incentive map of crypto](/blog/trading/crypto-players/cui-bono-the-incentive-map-of-crypto)
