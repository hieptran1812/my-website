---
title: "Why Markets Are Regulated: Disclosure and the Securities Acts"
date: "2026-06-21"
publishDate: "2026-06-21"
description: "Why securities markets need rules at all, why the US chose disclosure over judging investments, and how the 1933 and 1934 Acts built the trust that makes markets liquid."
tags: ["capital-markets", "securities-regulation", "disclosure", "securities-act-1933", "exchange-act-1934", "sec", "ipo", "reg-d", "sarbanes-oxley"]
category: "trading"
subcategory: "Capital Markets"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — Securities regulation exists to manufacture the one thing a market cannot run without: trust between strangers. The US does it by forcing **disclosure**, not by judging whether an investment is good.
>
> - Sellers know more than buyers. Left alone, that information gap collapses markets into a "lemons" trade where only bad deals get done — so the law forces issuers to publish the facts.
> - The US picked **disclosure over merit review**: regulators make companies tell the truth and let investors decide, rather than ruling on whether a stock is worth buying.
> - The **1933 Act** governs new issues (register, publish a prospectus); the **1934 Act** governs trading (it created the SEC, ongoing reporting, and the anti-fraud rules).
> - The number to remember: a US IPO costs **several million dollars** in legal, audit, and ongoing compliance — which is exactly why companies stay private longer and why **Reg D** private placements exist.

On the morning of December 11, 2020, Airbnb's shares opened on the Nasdaq at roughly \$146, more than double the \$68 price the company had sold them at the night before. Thousands of people who had never met the founders, never seen the books, never set foot in the company's offices wired billions of dollars into a business that had lost money for most of its life and had just survived a pandemic that nearly killed travel entirely.

Why would anyone do that? Not because they trusted Airbnb's management personally — they had never met them. They did it because, weeks earlier, Airbnb had filed a 350-page document with the US Securities and Exchange Commission that laid out, under penalty of law, exactly how much money it made, how much it lost, who its executives were, what could go wrong, and how its shares were structured. Auditors had signed the numbers. Lawyers had vetted the disclosures. The whole apparatus existed so that a stranger in Ohio could buy a slice of a California company with the same confidence as the venture capitalists who had spent years inside it.

That apparatus has a name: **disclosure-based securities regulation**. It is the least glamorous part of a capital market and the most important. This post is about why it exists, what it actually does, and how a handful of Depression-era laws built the trust that lets the entire machine run.

![Pipeline showing information gap to disclosure to trust to liquidity to issuance](/imgs/blogs/why-markets-are-regulated-disclosure-and-the-securities-acts-1.png)

## Foundations: why a market between strangers needs rules

Start with the spine of this whole series. A capital market is a **machine that turns savings into long-term investment**. Households and institutions have money they don't need today; companies and governments have projects that need funding now and will pay off later. The market connects the two. It runs on two engines: a **primary market**, where new securities are *created* to raise money (an IPO, a bond sale), and a **secondary market**, where those same securities are *traded* afterward so the original buyers can sell out (a stock exchange).

A **security** is simply a tradable claim on future cash flows — a share of a company's profits, a promise to repay a loan with interest. (We unpack exactly what counts as a security in [what a security actually is](/blog/trading/capital-markets/what-a-security-actually-is-claims-you-can-sell).) The defining feature is that it is *tradable*: you can sell your claim to a stranger.

And there's the problem. When you buy a used car from a friend, you can ask them, watch them drive it, trust their reputation. When you buy a thousand dollars of a company's stock, you are handing money to people you will never meet, in exchange for a piece of paper (now an electronic entry) that says you own a sliver of something you cannot inspect. The company's managers know everything about the business. You know almost nothing. That gap — between what insiders know and what outside investors know — is called **information asymmetry**, and it is the original sin of every securities market.

Here is why it's fatal if left alone. Suppose you're considering buying shares in a company, but you genuinely cannot tell whether it's a great business or a fraud. The seller knows which one it is; you don't. What price will you offer? Not the price of a great business — because it might be a fraud. Not the price of a fraud — because it might be great. You'll offer something in between, discounted heavily for the risk that you're being cheated. This is the **lemons problem**, named after the 1970 economics paper by George Akerlof about used cars ("lemons" being American slang for a defective car). When buyers can't distinguish good from bad, they price everything as if it might be bad.

Now watch what that discount does. The owners of genuinely good companies look at your lowball offer and refuse to sell — their business is worth far more than the suspicious price you're willing to pay. So they walk away. Only the owners of bad companies, who know they're getting a great deal at your discounted price, are happy to sell. The good issuers exit; the bad ones stay. The market fills up with exactly the deals you were afraid of, which makes buyers discount even harder, which drives out more good issuers. The market can unravel completely — not because there are no good companies, but because nobody can prove they're good.

Securities regulation is the fix for this unraveling. By **forcing every issuer to publish verified facts**, the law lets a good company *prove* it's good. The buyer can now price the facts instead of pricing their fear. Good issuers get a fair price and come to market; the lemons get exposed and priced accordingly. Trust — manufactured by mandatory disclosure — is what keeps the market from collapsing into a lemons trade. That is the entire game.

It's worth being clear about why the market can't just solve this itself, voluntarily. A good company *could* simply choose to publish its books to prove it's not a lemon — so why does the law have to force it? Two reasons. First, **credibility**: anyone can publish a glossy document claiming to be wonderful; what makes disclosure believable is that it's audited by independent accountants and backed by legal liability if it's false. Voluntary, unaudited self-praise carries no weight, because the lemons can write the same words. Second, **the standardization problem**: if every company disclosed whatever it wanted in whatever format it chose, investors couldn't compare them, and a clever lemon could disclose a flood of irrelevant detail while burying the one fact that matters. By *mandating* a standard set of disclosures — the same line items, audited the same way, in the same forms — the law makes companies comparable and makes omissions conspicuous. A missing risk factor in a standardized filing screams; a missing paragraph in a free-form brochure is invisible. Regulation isn't just demanding disclosure; it's demanding *credible, comparable* disclosure, which the market cannot reliably produce on its own.

## The lemons problem, in dollars

Before going further, let's put a dollar figure on the lemons discount, because it is not a metaphor — it's a number you can compute.

#### Worked example: the lemons discount on a \$100 share

Suppose a company's shares are genuinely worth \$100 each if the business is as good as management claims, and worth \$40 if it turns out the books are cooked. You, the outside buyer, think there's a 70% chance it's the good case and a 30% chance it's the bad case, and crucially **you cannot verify which**. A rational price is the probability-weighted value:

`(0.70 × \$100) + (0.30 × \$40) = \$70 + \$12 = \$82.`

So you'll pay \$82 for a share that, if honest, is worth \$100. That \$18 gap is the **lemons discount** — the tax the information gap charges every honest issuer. The good company's owners, who know it's truly worth \$100, look at your \$82 offer and many of them refuse to sell at all. The intuition: the information gap doesn't just lower prices, it *drives the best issuers out of the market entirely*.

#### Worked example: what disclosure is worth

Now the company files audited financials that credibly prove it's the good case. Your probability of the good outcome jumps from 70% to, say, 98%. The new fair price:

`(0.98 × \$100) + (0.02 × \$40) = \$98 + \$0.80 = \$98.80.`

Disclosure moved the price from \$82 to \$98.80 — almost \$17 per share recovered for the honest issuer. On a 10-million-share offering, that's nearly **\$170 million** of value that mandatory disclosure handed back to the company and its existing owners. The intuition: disclosure is not a cost the issuer grudgingly bears; for an honest company it's the thing that lets it capture its real value.

![Before and after comparison of a lemons market versus a disclosed market](/imgs/blogs/why-markets-are-regulated-disclosure-and-the-securities-acts-3.png)

This is why the spine of the series matters so much here. Secondary-market liquidity — the ability to sell your claim tomorrow morning — is what makes anyone willing to fund a 30-year project today. But liquidity only exists if strangers are willing to trade with each other, and they only do that if they trust the facts. **Disclosure manufactures the trust; trust creates the liquidity; liquidity makes issuance possible.** Pull out the disclosure layer and the whole machine seizes.

## The US philosophy: disclosure, not merit

Here is the choice that defines American securities law, and it is not the only choice a country could make.

Suppose you're a regulator and a company wants to sell stock to the public. You could do one of two things. You could read the filing and decide whether this is a *good investment* — block the offering if you think the company is overpriced or the business is weak. That's called **merit review**: the regulator passes judgment on the quality of the deal. Some US states used to do exactly this under old "blue sky" laws, and some countries still lean this way.

Or you could take a completely different stance: you don't judge whether the investment is good. You only insist that the company **tell the truth, fully and clearly**, and then you let investors decide for themselves. That's **disclosure-based regulation**, and it's the path US federal law chose in 1933.

The slogan comes from Supreme Court Justice Louis Brandeis, who wrote in 1914 that "sunlight is said to be the best of disinfectants." The idea: you don't need a government official to decide which companies are worthy. You need to force every company into the sunlight — publish the facts — and the collective judgment of millions of investors will price them. A terrible company can absolutely sell stock to the public in the US, as long as it honestly discloses that it's terrible. The SEC's job is not to stop you from making a bad investment. It's to make sure you can't claim afterward that you weren't told the risks.

This distinction is load-bearing, and beginners get it wrong constantly. The SEC does **not** "approve" securities. When a prospectus says the offering has been registered with the SEC, that means the disclosure was reviewed for completeness and form — not that the government thinks it's a good buy. Every US prospectus is legally required to say, in effect, that the SEC has not approved or disapproved of the securities and has not judged whether the disclosure is truthful. The regulator builds the sunlight; you decide whether to stand in it.

Why did the US go this way? Partly philosophy — a distrust of putting a bureaucrat between citizens and their own financial choices. Partly practicality — no regulator can reliably judge which of ten thousand companies will succeed, but any regulator can check whether a company published audited numbers and disclosed its risks. Disclosure scales; merit review doesn't. The cost is that disclosure-based regimes let bad investments happen in plain sight. The benefit is a vast, deep, liquid market where capital flows to wherever investors collectively think it belongs.

## The 1933 Act: truth in the primary market

The two foundational US laws split the work cleanly along the primary/secondary line of this series.

The **Securities Act of 1933** governs the **primary market** — the moment a security is *created and sold to the public for the first time*. Its core demand is simple: before you sell securities to the public, you must **register** them with the SEC and deliver a **prospectus** to buyers. The 1933 Act was Congress's direct response to the 1929 crash, when countless Americans had bought stocks on the strength of rumor, tips, and outright fraud, with no reliable information at all.

Registration means filing a detailed document — for a stock IPO, this is the **Form S-1** — that lays out the business, the financials (audited), the management, the risk factors, what the company will do with the money, and the terms of the securities. The **prospectus** is the part of that filing handed to investors. It is the legal embodiment of "sunlight": a standardized, audited, liability-backed account of what you're buying. We go deep on the prospectus itself in [disclosure: the prospectus, filings, and insider trading](/blog/trading/capital-markets/disclosure-the-prospectus-filings-and-insider-trading); here the point is that the 1933 Act is what *forces* the prospectus to exist.

The teeth of the 1933 Act are in **liability**. If the registration statement contains a material misstatement or omits something it should have disclosed, the company, its directors, its underwriters, and the accountants who signed the numbers can all be sued by burned investors. That liability is why underwriters perform exhausting **due diligence** before an IPO and why auditors sign carefully — they're personally on the hook. Disclosure isn't trustworthy because companies are honest; it's trustworthy because lying is expensive.

#### Worked example: what a \$500M IPO costs to disclose

Say a company runs a \$500 million IPO. Direct costs of getting through the 1933 Act regime and to market typically include: underwriting fees (the banks' cut, often around 5–7% for a deal this size, so roughly \$25–35 million), plus legal fees of \$2–4 million, audit and accounting fees of \$1–3 million, SEC registration and exchange listing fees, printing, and the cost of the roadshow. The non-underwriting "cost of being public-ready" — legal, audit, financial-reporting build-out — commonly runs **\$4–8 million** before the underwriting spread. Then, every year after, ongoing public-company compliance (audits, quarterly reporting, the controls work below) adds **\$2–5 million annually**. The intuition: registering with the public is not a one-time fee — it's a permanent operating cost the company signs up for in exchange for access to public capital.

## The 1934 Act: the SEC and the ongoing rules of trading

The **Securities Exchange Act of 1934** governs the **secondary market** — the continuous *trading* of securities after they've been issued. If the 1933 Act is about the moment of sale, the 1934 Act is about everything that happens for the rest of the security's life. It does several enormous things.

First, it **created the Securities and Exchange Commission** — the SEC — as the federal agency that administers and enforces the securities laws. Before 1934 there was no single federal securities regulator at all. Second, it established **ongoing reporting**: a public company doesn't disclose once at IPO and then go dark. It must file an annual report (**Form 10-K**), quarterly reports (**Form 10-Q**), and prompt reports of major events (**Form 8-K**), forever, as long as its shares trade publicly. The sunlight has to keep shining, not flash once. Third, it created the **anti-fraud rules** — most famously Rule 10b-5, the catch-all prohibition on fraud and manipulation in connection with buying or selling securities, which is the legal basis for most insider-trading and market-manipulation cases.

The 1934 Act also brought the exchanges and brokers themselves under federal oversight, requiring them to register and submit to rules. The mechanics of how exchanges and clearinghouses actually operate under this regime are covered in [stock exchanges and clearinghouses](/blog/trading/finance/stock-exchanges-and-clearinghouses), and the specific machinery of keeping trading honest — manipulation, spoofing, circuit breakers — is the subject of [market integrity](/blog/trading/capital-markets/market-integrity-manipulation-spoofing-and-circuit-breakers).

The reason ongoing reporting matters so much for our spine is the secondary market. A stranger buying your Airbnb shares three years after the IPO needs current facts, not a snapshot from listing day. The 10-K and 10-Q keep the information flowing so that the security stays priceable — and therefore stays liquid — for its entire life. Without continuous disclosure, every security would slowly drift back toward the lemons problem as its IPO-era information went stale.

It's worth pausing on *why* the law splits the job into two acts rather than one. The risks are genuinely different. At the moment of issuance — the 1933 Act's domain — the danger is the cold-start problem: there is no trading history, no track record of the security in public hands, nothing for a buyer to lean on except what the issuer chooses to tell them. So the law front-loads an exhaustive, one-time, liability-backed disclosure: the full registration statement. Once the security is trading — the 1934 Act's domain — a market price exists and updates every second, and the risk shifts to *staleness* and *abuse*: management letting the facts go dark, or insiders trading on what they know before the public does. So the law switches from a one-time blast of disclosure to a steady drip of periodic reports plus anti-fraud rules that police the trading itself. Same goal, manufacturing trust, but two different failure modes demanding two different tools.

### What the SEC actually does

It helps to be concrete about the agency the 1934 Act created, because "the SEC" gets invoked vaguely. The Commission does three distinct kinds of work, and conflating them is a common source of confusion.

First, **rulemaking**. Congress writes statutes in broad strokes; the SEC fills in the operational detail through rules. When Congress said "no fraud in connection with securities," the SEC wrote Rule 10b-5 to spell out what that means. When it decided private placements needed clear boundaries, the SEC wrote Regulation D. The rulebook is enormous and constantly evolving, and most of the day-to-day texture of how markets operate lives in SEC rules, not in the original acts.

Second, **review and oversight**. The SEC's Division of Corporation Finance reviews registration statements and periodic filings for completeness and compliance — checking that a prospectus discloses what the rules require, issuing comment letters demanding more detail where a filing is thin. Crucially, this review is about *adequacy of disclosure*, never about whether the investment is good. A company can clear SEC review with flying colors and still be a terrible business — the review only certifies that the terrible-ness was properly disclosed.

Third, **enforcement**. When someone breaks the rules — an insider who traded on secret information, an executive who signed off on cooked numbers, a fund manager who lied to clients — the SEC's Division of Enforcement investigates and brings cases, seeking fines, disgorgement of ill-gotten gains, bars from serving as a public-company officer, and referrals for criminal prosecution. Enforcement is what gives disclosure its teeth. A disclosure requirement that nobody enforced would be theater; the credible threat of an enforcement action is what makes issuers tell the truth in the first place. The deterrent, not the paperwork, is the product.

#### Worked example: the enforcement deterrent in dollars

Consider an executive sitting on material non-public news — say, that next quarter's earnings will miss badly. They hold \$2 million of company stock that would drop, and could quietly sell before the announcement to dodge a \$400,000 loss. Without enforcement, that's a free \$400,000. With enforcement, the expected math flips: if the SEC catches insider trades with even a 30% probability and the penalty is disgorgement of the \$400,000 *plus* a civil penalty of up to three times that (\$1.2 million) plus the risk of a criminal case and a career-ending officer-and-director bar, the expected cost of trading is roughly `0.30 × (\$400,000 + \$1,200,000) = \$480,000` — already more than the \$400,000 saved, before counting prison risk. The intuition: enforcement doesn't have to catch everyone; it just has to make the *expected* cost of cheating exceed the gain, which turns the disclosure rules from suggestions into constraints people actually obey.

![Timeline of US securities laws from 1933 through the JOBS Act](/imgs/blogs/why-markets-are-regulated-disclosure-and-the-securities-acts-4.png)

## The alphabet of later laws

The 1933 and 1934 Acts are the foundation, but US securities regulation grew by accretion — each major failure produced a new layer bolted onto the original two. You don't need to memorize them, but you should recognize the pattern: a crisis exposes a gap, Congress legislates the gap shut.

**Glass-Steagall (1933)** separated commercial banking (taking deposits, making loans) from investment banking (underwriting and trading securities), after the crash made lawmakers fear that banks gambling with depositor money was dangerous. Its core separation was largely repealed in 1999 — a decision much debated after 2008.

The **Investment Company Act and Investment Advisers Act (1940)** brought mutual funds and the people who manage other people's money under regulation, requiring disclosure and limiting conflicts of interest. (How the buy-side firms themselves operate is a separate world — see the dedicated hedge-funds and asset-management material rather than re-deriving it here.)

**Sarbanes-Oxley (2002)**, usually called SOX, was the response to the Enron and WorldCom accounting frauds, where companies had filed disclosures that were thoroughly cooked. SOX made the CEO and CFO **personally certify** the financial statements under criminal penalty, created the **PCAOB** to oversee the auditors who had been complicit, and required companies to document and test their internal financial controls (the infamous Section 404). The lesson SOX encodes: disclosure is only as good as the controls behind it.

**Dodd-Frank (2010)** was the sprawling response to the 2008 financial crisis, pushing huge parts of the previously dark derivatives market into clearing and reporting, creating new systemic-risk oversight, and tightening rules on the institutions whose failure could take down the system.

The **JOBS Act (2012)** went the *other* direction — it *eased* disclosure burdens for smaller and newly public companies, on the theory that the full 1933/1934 apparatus was so expensive it was choking off small-company access to capital. It's the clearest official acknowledgment that disclosure has a cost worth managing, not just a benefit.

#### Worked example: the SOX bill for a mid-cap

Take a company with \$800 million in revenue going public, or freshly public. Sarbanes-Oxley Section 404 requires it to document, test, and have its auditor attest to its internal financial controls. For a mid-cap, the first-year build-out of SOX compliance — extra audit fees, internal-controls consultants, new finance hires, software — commonly runs **\$1.5–3 million**, with ongoing annual costs of perhaps **\$1–2 million** thereafter. Spread over, say, \$800 million of revenue that's a fraction of a percent — survivable for a real business. But for a tiny company with \$30 million of revenue, a \$2 million compliance bill is over 6% of revenue. The intuition: the *fixed* cost of disclosure falls hardest on small issuers, which is exactly the gap the JOBS Act tried to close.

## Registration versus exemption: the central trade-off

Now the practical question every company and dealmaker faces: do you go through the full public registration, or do you find an **exemption** that lets you skip it?

Registering under the 1933 Act and reporting under the 1934 Act is the path to **public capital**: anyone in the world can buy your securities, they trade freely on an exchange, and you tap the deepest pools of money in existence. The price is total disclosure, forever, with all the cost and liability that entails.

But the law also carves out **exemptions** — ways to sell securities *without* full public registration, on the logic that some buyers don't need the protection of mandatory disclosure because they can fend for themselves. The most important is **Regulation D (Reg D)**, which lets a company raise money privately from **accredited investors** — broadly, people or institutions wealthy or sophisticated enough that the law presumes they can demand information themselves and absorb losses. This is how almost all startup venture funding, private equity, and private credit gets done. No S-1, no prospectus, no SEC registration of the offering — just a much lighter filing and a sale to qualified buyers.

The catch is the flip side of the spine. Securities sold under a Reg D exemption are **restricted** — they can't be freely resold to the public. The buyer gives up the liquid secondary market in exchange for getting in early and cheap. So the trade-off is exact: **public registration buys liquidity and capital access at the price of disclosure cost; private exemption avoids the cost but forfeits the liquidity.**

![Matrix comparing a registered IPO with a Reg D private placement](/imgs/blogs/why-markets-are-regulated-disclosure-and-the-securities-acts-6.png)

#### Worked example: a \$40M Reg D private placement

A growing software company needs \$40 million but isn't ready for the cost and scrutiny of an IPO. It runs a **Reg D private placement**: it sells \$40 million of preferred stock to a dozen venture funds and a few wealthy individuals, all accredited investors. Legal cost might be \$150,000–400,000 — versus the \$4–8 million-plus a registered IPO would have cost — and there's no public prospectus, no S-1, no quarterly reporting to the SEC. In exchange, those investors hold **restricted** shares: they can't dump them on the public market and generally have to wait for an eventual IPO or acquisition to cash out. The intuition: Reg D lets a company raise serious money cheaply, but it pushes the liquidity problem down the road — the investors are betting on a future exit that doesn't yet exist.

## The cost side: why companies stay private longer

For decades, "going public" was the natural destiny of a successful company. That has changed, and disclosure cost is a big part of why. The combination of SOX-era compliance burden, the litigation exposure of being a public reporting company, and the sheer availability of enormous private capital (huge venture and private-equity funds happy to do Reg D rounds) has made staying private far more attractive than it once was.

The numbers tell the story. The total **count** of US IPOs has trended down over the long run even as the economy grew, and the number of US public companies is well below its late-1990s peak. Companies that once would have gone public at a \$500 million valuation now raise multiple private rounds and list — if at all — at \$10 billion or more, having captured most of their growth while private. The disclosure regime that protects public investors also, by being costly, keeps companies away from public investors for longer.

![Number of US IPOs per year from 2014 to 2024](/imgs/blogs/why-markets-are-regulated-disclosure-and-the-securities-acts-7.png)

This is a genuine policy tension with no clean answer. Heavy disclosure protects the small public investor and keeps the public market trustworthy and liquid. But it also means ordinary investors are increasingly locked out of a company's highest-growth years — those gains now accrue to the accredited investors in the private rounds. The trust factory has a side effect: by making the public market safe, it makes the public market expensive to enter, which pushes value creation into the private market that ordinary savers can't reach.

There's a second, subtler cost that doesn't show up in the dollar bill at all: the **distortion of behavior** that mandatory disclosure can create. When a company must report earnings every quarter to the public, its managers face relentless pressure to hit short-term numbers, because a single weak quarter can crater the stock and invite lawsuits. Some executives explicitly cite this "quarterly treadmill" as a reason to stay private — they'd rather invest for a five-year horizon than explain a soft quarter to traders who will sell first and read the footnotes later. Disclosure makes the market trustworthy, but the same transparency that builds trust can also push public-company management toward myopia. It's another reason the calculus of going public has tilted: the cost of disclosure is not only the audit fees, but the loss of the patience that private ownership can afford.

#### Worked example: the threshold where staying private wins

Take a company weighing an IPO. Going public might raise its valuation multiple — public investors often pay a premium for liquidity — say lifting a \$2 billion private valuation to \$2.4 billion, a \$400 million "liquidity bump." Against that, set the costs: ~\$30 million in IPO underwriting and fees up front, then ~\$4 million a year in ongoing public-company compliance and the harder-to-quantify drag of the quarterly treadmill. If the company can instead raise the capital it needs through a Reg D round at, say, a \$2.1 billion valuation with negligible disclosure cost, the comparison is between a one-time \$300 million valuation gain (net of fees) that comes with a permanent \$4M/yr cost and behavioral drag, versus staying private cheaply. For a company with abundant private capital available, the \$4M/yr-plus-treadmill side increasingly loses. The intuition: as private capital got deep and cheap, the disclosure cost of being public stopped being a rounding error and started tipping real decisions toward staying private.

## Common misconceptions

**"The SEC approves stocks before you can buy them."** No. The SEC reviews disclosure for completeness and compliance; it does not pass judgment on whether a security is a good investment. A company can sell shares in a money-losing, high-risk business as long as it discloses those facts honestly. The legal boilerplate on every prospectus explicitly disclaims any government endorsement.

**"Regulation is just a tax on business — markets would be fine without it."** The lemons problem says otherwise. Without credible disclosure, the *good* issuers are the ones driven out, because they refuse to sell at the discount buyers demand for unverifiable claims. The math in the worked examples is real: an honest issuer can recover something like \$17 per share — \$170 million on a 10-million-share float — purely from being able to prove its quality. Disclosure is what lets honest companies get a fair price.

**"Private placements are a loophole that skips the rules."** They're not a loophole — they're a deliberate, codified exemption (Reg D) built on the logic that sophisticated, wealthy buyers don't need the same protection as the retail public. And the exemption carries a real price: the securities are restricted and illiquid. You don't get the protection *and* the liquid market; you trade one for the other.

**"The 1933 and 1934 Acts are old and basically irrelevant now."** They are the living spine of every public offering and every trade today. The S-1 Airbnb filed, the 10-Ks every public company files quarterly, the insider-trading cases in the news, the very existence of the SEC — all of it flows directly from those two Depression-era statutes. Everything since (SOX, Dodd-Frank, JOBS) is an amendment to that foundation, not a replacement.

**"More disclosure is always better."** The JOBS Act exists precisely because Congress concluded that wasn't true. Past some point, disclosure cost chokes off small-company access to capital and pushes companies to stay private, locking ordinary investors out of growth. Good regulation is a balance between trust and cost, not a maximization of paperwork.

## How it shows up in real markets

**The 2021 boom and the 2022 collapse.** In 2021, with cheap money and roaring confidence, US companies raised around \$142 billion in traditional IPOs across nearly 400 deals. In 2022, as interest rates spiked and confidence cratered, IPO proceeds collapsed to roughly \$8 billion — a fall of more than 90% — with the deal count dropping from ~397 to ~71. The companies didn't suddenly become worse; the disclosure regime didn't change. What collapsed was *trust* in valuations and the willingness of buyers to fund new issues. It's the cleanest illustration of the spine: when confidence evaporates, the primary market — issuance — freezes, even though the secondary market keeps trading. Issuance is the part that needs trust most.

![US IPO proceeds by year showing the 2021 peak and 2022 collapse](/imgs/blogs/why-markets-are-regulated-disclosure-and-the-securities-acts-2.png)

**Enron and the birth of SOX.** Enron filed audited financials that the disclosure regime was supposed to make trustworthy — and they were fiction, hiding billions in debt in off-balance-sheet vehicles while the auditor, Arthur Andersen, signed off. The fraud destroyed the company in 2001 and wiped out employees' retirement savings. The lesson the regulators drew was that disclosure is worthless if the controls and the auditors behind it can be captured. SOX (2002) was the answer: personal CEO/CFO certification under criminal penalty, an independent regulator for auditors, and mandatory internal-controls testing. It's a direct illustration that the disclosure regime is not static — it gets re-engineered every time a failure shows how the trust can be faked.

**The long-run trust dividend.** Step back from any single year and look at the stock of capital the system supports. US equity market cap grew from roughly \$26 trillion in 2014 to around \$58 trillion by 2024. That stock of value exists because tens of millions of strangers are willing to hold and trade claims on companies they'll never inspect — which they only do because the disclosure regime makes the claims trustworthy and the market liquid. The chart below is, in a real sense, a picture of accumulated trust.

![US equity market capitalization from 2014 to 2024](/imgs/blogs/why-markets-are-regulated-disclosure-and-the-securities-acts-5.png)

**The global scale of what's at stake.** Disclosure-based regulation isn't a niche American quirk; it underwrites the largest pools of capital on earth. Global equity market cap runs around \$115 trillion and the global bond market around \$140 trillion, with the US alone accounting for roughly \$55 trillion of equities and a similar amount of bonds. Different jurisdictions implement the trust factory differently — the EU's MiFID and Reg NMS in the US structure trading in their own ways, covered in [global market structure](/blog/trading/capital-markets/global-market-structure-reg-nms-mifid-and-cross-border) — but every deep, liquid market on the planet rests on some version of the same bargain: force out the facts, and strangers will fund the future.

![Global and US equity and bond market sizes in trillions of dollars](/imgs/blogs/why-markets-are-regulated-disclosure-and-the-securities-acts-8.png)

## The takeaway: regulation is the trust factory

The instinct most people have about securities regulation is that it's a brake — a set of rules that slow markets down and tax companies for the privilege of operating. That instinct has the causality backwards. Disclosure-based regulation isn't a brake on the market; it's the **engine of trust that lets the market exist at all**.

Trace the chain one more time. Strangers won't fund a company they can't verify, because the lemons problem forces them to assume the worst and price accordingly — which drives the good companies out and collapses the market. Mandatory disclosure breaks that trap: it lets honest issuers prove their quality, so buyers price the facts instead of their fear. That trust is what makes investors willing to trade, which makes the secondary market liquid, which is the *precondition* for anyone to buy into a primary issuance in the first place. Nobody funds a 30-year project unless they believe they can sell their claim on it tomorrow morning — and that belief is manufactured, brick by brick, out of S-1s and 10-Ks and the credible threat of an SEC enforcement action.

The US chose to build that trust through sunlight rather than judgment — disclose everything and let investors decide, rather than having a regulator rule on what's worthy. The cost is real and rising: several million dollars per IPO and per year, enough that companies increasingly stay private and ordinary investors get locked out of the best growth. The benefit is the deepest, most liquid capital markets in history. When you understand that regulation is the trust factory — not a tax, but the thing that converts an information gap into a working market — the rest of the machine snaps into focus. Every disclosure rule, every filing requirement, every enforcement case is in service of one job: making it safe for a stranger to buy what another stranger is selling.

## Further reading & cross-links

- [What a security actually is: claims you can sell](/blog/trading/capital-markets/what-a-security-actually-is-claims-you-can-sell) — the building block this whole regime regulates.
- [Disclosure: the prospectus, filings, and insider trading](/blog/trading/capital-markets/disclosure-the-prospectus-filings-and-insider-trading) — the next step: what the mandated documents actually contain and how insider-trading law works.
- [Market integrity: manipulation, spoofing, and circuit breakers](/blog/trading/capital-markets/market-integrity-manipulation-spoofing-and-circuit-breakers) — keeping the *trading* honest, the other half of trust.
- [The IPO process, end to end: from mandate to first trade](/blog/trading/capital-markets/the-ipo-process-end-to-end-from-mandate-to-first-trade) — how a company actually navigates the 1933 Act in practice.
- [Global market structure: Reg NMS, MiFID, and cross-border](/blog/trading/capital-markets/global-market-structure-reg-nms-mifid-and-cross-border) — how other jurisdictions build their own trust factories.
- [Stock exchanges and clearinghouses](/blog/trading/finance/stock-exchanges-and-clearinghouses) — the venues and plumbing that operate under the 1934 Act.
