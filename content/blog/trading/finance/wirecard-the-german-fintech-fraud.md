---
title: "Wirecard: The German Fintech Star That Was Missing 1.9 Billion Euros"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "A from-scratch walkthrough of how a celebrated German fintech reported profits and cash that did not exist, why its auditor and regulator missed it for years, and how a handful of journalists and short sellers were proven right."
tags: ["wirecard", "accounting-fraud", "fintech", "payment-processing", "auditing", "bafin", "short-selling", "case-study", "corporate-governance", "financial-statements"]
category: "trading"
subcategory: "Finance"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — Wirecard was Germany's celebrated fintech champion whose enormous Asian profits and 1.9 billion euros of trustee cash turned out to be largely invented, a fraud that survived for years because its auditor never confirmed the cash, its regulator attacked the journalists exposing it, and almost no one checked whether the money was actually there.
>
> - Wirecard processed card payments, but the profitable, fast-growing part of the business it showed investors ran through opaque Asian "third-party acquirer" partners whose merchants and fees may never have existed.
> - The reported profits were supposedly turned into cash held by trustees in escrow accounts; on June 18, 2020, roughly 1.9 billion euros (about \$2.1 billion) of that cash was found never to have existed.
> - The Financial Times and short sellers had flagged the numbers for years; Germany's regulator BaFin responded by investigating the journalists and short sellers and briefly banning short-selling of the stock, while the auditor EY kept signing the accounts.
> - The stock fell from about 104 euros to near zero in days; CEO Markus Braun was arrested, COO Jan Marsalek fled and vanished, and Wirecard filed for insolvency on June 25, 2020.
> - The durable lesson: a public listing, a clean audit, and a national regulator's backing are not proof that the cash exists; someone has to actually confirm the bank balance.

In early 2018, Wirecard was added to Germany's DAX-30, the index of the country's thirty most valuable public companies, pushing aside the 150-year-old Commerzbank to take its place beside Siemens, BMW, and SAP. At its peak the company was worth roughly 24 billion euros — about \$28 billion — more than Deutsche Bank, the country's largest lender. It was hailed as proof that Europe could grow its own technology giant, a fintech that had supposedly cracked the unglamorous business of moving money and turned it into a soaring growth story. Two and a half years later, on June 25, 2020, Wirecard filed for insolvency, the first member of the DAX ever to do so. The reason was simple and almost unbelievable: a large share of the company's reported profits, and 1.9 billion euros of cash that was supposed to be sitting safely in Asian bank accounts, did not exist.

This is the story of how a company can join a national stock index, pass its audits for a decade, command the protection of its national regulator, and be celebrated as a technology champion while being, underneath, a partly fictional enterprise. It is not a story about a clever derivative or a market crash. It is a story about a few deceptively simple questions — *who are these customers? where is the cash? has anyone actually checked?* — that almost no one with power asked until it was far too late.

The diagram above is the mental model: a long climb built on profits booked through opaque partners and cash that supposedly piled up in trustee accounts, followed by a collapse that took only days once someone finally asked the bank whether the money was there.

![Timeline of Wirecard from its 1999 founding through its 2020 collapse and the criminal trials that followed](/imgs/blogs/wirecard-the-german-fintech-fraud-1.png)

We are going to build this up from zero. If you have never thought about how a card payment actually works, that is fine. By the end you will understand what a payment processor does and how it earns its fees; what a "third-party acquirer" arrangement is and why it is hard to verify; what an escrow account and a trustee are; how revenue and cash are supposed to be recognised and confirmed; what an external auditor is actually obligated to check; why short sellers and investigative journalists are the market's professional skeptics; and how a public listing lends a company a credibility it may not deserve. Then we will watch all of it break.

## Foundations: how moving money is supposed to work

Before we can see the trick, we need to see the honest version. Wirecard's real business sat inside the global plumbing that moves money every time you tap a card. So let us start there.

### What a payment processor actually does

When you buy a coffee for \$5 with a credit card, a small chain of companies springs into action in under two seconds. Your bank — the **issuing bank**, or *issuer*, the one that gave you the card — has to confirm you have the money or credit. The coffee shop's bank — the **acquiring bank**, or *acquirer*, the one that accepts payments on the merchant's behalf — has to receive the funds. Between them sits a **card network** like Visa or Mastercard, which routes the request and sets the rules. And somewhere in the middle sits a **payment processor**: the technical company that takes the card details from the terminal, packages them into the right messages, sends them to the network, handles the approval or decline, and later makes sure the right amount of money lands in the merchant's account.

For doing this, the processor earns a fee. On that \$5 coffee, the merchant might pay roughly 2% to 3% in total fees, most of which is split among the issuer, the network, and the acquirer/processor. The processor's own slice might be a fraction of a percent — perhaps a cent or two per transaction. It sounds tiny, and it is, on one coffee. But the business is built on volume. Process billions of transactions a year and those fractions of a cent add up to hundreds of millions in revenue. This is a real, legitimate, and large business. Companies like Stripe, Adyen, PayPal, and the acquiring arms of big banks make serious money doing exactly this, and Wirecard's *European* operations did a version of it too. The reader who wants the honest, thriving version of this industry can see how [the modern fintech disruptors built it](/blog/trading/finance/fintech-disruptors-stripe-paypal-ant).

The key thing to hold onto: a payment processor's revenue is *transaction fees*, and those fees come from *real merchants processing real payments from real customers*. If the merchants are fake, or the transactions never happened, the fees are fiction — and so is everything built on top of them.

There is one more wrinkle worth knowing, because it shaped Wirecard's origins. Not all merchants are equal in the eyes of a payment processor. Some carry a much higher rate of **chargebacks** — transactions a customer later disputes and claws back, often because the goods never arrived, the charge was fraudulent, or the customer simply denies it. Online gambling, adult content, and certain subscription businesses have notoriously high chargeback rates. Mainstream banks and processors often refuse these "high-risk" merchants, because every chargeback can cost the processor money and because regulators scrutinise the categories. A processor willing to serve high-risk merchants can charge them much higher fees — sometimes several times the rate of an ordinary shop — to compensate for the risk. That higher-fee, higher-risk niche is a legitimate corner of the industry, and it is where Wirecard began. It is also a corner with weaker scrutiny and more opaque flows, which is relevant to how a fraud could later hide inside payment data.

### Third-party acquirers: outsourcing the part you cannot see

Here is the first concept that the whole Wirecard fraud turns on. A payment processor cannot always operate directly in every country. To process payments in a particular region, you often need local banking licences, local relationships, and local approval from the card networks. Rather than build all of that itself in, say, Southeast Asia or the Middle East, a processor can outsource it to a local partner called a **third-party acquirer**, or **TPA**.

In a TPA arrangement, the local partner does the actual processing for merchants in its region, and the Western processor — Wirecard, in our story — books a share of the resulting fees as its own revenue, paying the partner for the access and the local work. On paper this is a normal, defensible structure. Plenty of legitimate companies use local partners to reach markets they cannot enter directly.

But notice what it does to *visibility*. The merchants are the partner's merchants. The transactions run on the partner's systems. The contracts, the customer lists, the actual flow of payments — all of it sits one company removed from the firm reporting the revenue. If Wirecard claims that a TPA partner in Asia generated, say, \$300 million of processing volume last quarter and that Wirecard's cut was \$90 million of profit, the only people who can directly confirm those merchants and those transactions are the partner and the merchants themselves. The reported number is real only if someone independent actually checks the underlying business. That gap between *what is reported* and *what can be independently verified* is exactly where a fraud can live.

### Escrow accounts and trustees: where the cash is supposed to sit

The next concept is the **escrow account**. An escrow account is a holding account controlled by a neutral third party — a **trustee** (often a law firm or a specialised escrow agent) — who holds money on behalf of others and releases it only under agreed conditions. Escrow is everywhere in legitimate finance: when you buy a house, your deposit often sits in escrow until closing; the trustee's whole job is to be a trustworthy in-between so neither side has to trust the other.

In Wirecard's TPA business, the story it told was this: because the partners handled the processing, Wirecard's share of the profits accumulated in escrow accounts held by trustees, to be released to Wirecard later. So a large and growing pile of Wirecard's reported cash was not sitting in Wirecard's own bank account in Germany. It was supposedly sitting in **trustee-controlled escrow accounts in Asia**, holding the accumulated profits of the third-party-acquirer business.

This matters enormously, because *cash* is the one number on a company's books that is supposed to be the easiest and hardest to fake at the same time. Easiest, because confirming cash is a simple matter in principle: ask the bank holding it to confirm the balance directly. Hardest, because if you can manufacture a convincing confirmation, you can make a number that everyone treats as bedrock-solid into pure fiction. We will see exactly how that played out.

There is a subtle reason fraudsters favour an escrow-and-trustee structure for hiding fictional cash, and it is worth naming. A normal bank account is confirmed by going to the bank. But an escrow arrangement inserts a *trustee* between the company and the bank, and it invites everyone to rely on the trustee's word — the trustee's statements, the trustee's confirmations — rather than on the underlying bank directly. If you can control or fool the trustee, or fabricate documents that appear to come from the trustee and the bank, you have built a layer of plausible deniability around a number that does not exist. The very feature that makes escrow trustworthy in honest hands — a neutral party vouching for the money — becomes the stage for the deception when the vouching is faked. That inversion is the heart of what went wrong here.

### Revenue, profit, and the gap with cash

A quick refresher on the numbers a company reports, because the fraud lived in the difference between them. **Revenue** is the total money a company bills for its services — here, processing fees. **Profit** (or **net income**, or **earnings**) is what is left after costs. And **cash** is the actual money the company holds. In an honest company these track each other over time: you earn revenue, you collect cash, the cash shows up in the bank.

The single most diagnostic question in nearly every accounting scandal is: *does the reported profit turn into real cash?* A company can report enormous profits, but if the cash never actually arrives — or arrives only as a number in an account no one has independently confirmed — then the profit is, at best, a promise and, at worst, an invention. We will return to this again and again, because it is the thread that unravels Wirecard.

### The external auditor: the referee who is supposed to confirm the cash

A public company's financial statements are checked by an outside firm called an **external auditor** — an accounting firm hired to examine the books and issue an opinion on whether the statements are fair and follow the rules. The auditor's signature is supposed to be the investor's protection: an independent professional saying, "We looked; these numbers are reliable." Wirecard's auditor was **Ernst & Young** (**EY**), one of the "Big Four" global accounting firms, and EY signed off on Wirecard's accounts year after year for roughly a decade.

Here is the part that will matter most. One of an auditor's most basic, almost mechanical duties is to **independently confirm cash balances**. The standard procedure is called a *bank confirmation*: the auditor sends a request *directly to the bank* — not through the client, not relying on documents the client hands over — and the bank replies *directly to the auditor* confirming how much money is in the account. It is one of the oldest and most fundamental checks in the entire profession, taught in the first weeks of audit training, precisely because cash is the asset most worth confirming and the easiest to confirm. As we will see, for the most important balances in this story, that direct confirmation either was not properly done or relied on documents that turned out to be forged.

### Short sellers and journalists: the people paid to be skeptical

Most investors make money when a stock rises. A **short seller** does the opposite. To short a stock, you borrow shares from someone who owns them, sell them at today's price, and promise to buy them back later to return them. If the price falls, you buy back cheaper and pocket the difference; if it rises, you lose, and your losses can be very large. Because a short seller profits from a company's *failure*, they have a powerful incentive to dig for problems everyone else is ignoring or unwilling to see. They are the market's professional skeptics. Several short sellers spent years betting that Wirecard's numbers were false.

Alongside them were **investigative journalists**, most importantly **Dan McCrum** at the *Financial Times*, who spent years reporting on irregularities in Wirecard's accounts. Journalists and short sellers are not the same — a reporter is not betting on the stock — but in this story they played complementary roles as the people willing to ask the uncomfortable questions. And both became targets.

### How a public listing confers credibility it cannot guarantee

One last foundational idea. When a company is publicly listed — its shares traded on a stock exchange — it gains an enormous amount of *implied* credibility. The reasoning that washes over most observers goes: it is listed on a major exchange, it is in the national index, it is audited by a Big Four firm, it is overseen by the national financial regulator, large institutional investors own it, banks lend to it — surely all of those sophisticated gatekeepers cannot all be wrong. This *halo* is real and powerful, and it is exactly what let Wirecard's claims go unchallenged for so long. The uncomfortable truth this case teaches is that the halo is a chain of assumptions, and a chain is only as strong as whether anyone in it actually did the checking.

With those building blocks in place — processors and TPAs, escrow and trustees, revenue versus cash, the auditor's duty to confirm cash, the skeptics, and the credibility halo — we can now watch the machine get built.

## The setup: from porn and gambling payments to a DAX-30 champion

Wirecard was founded in 1999, near the height of the first dot-com boom, as a company that processed online payments. In its early years its bread-and-butter customers were precisely the merchants that mainstream banks were reluctant to touch: online pornography and gambling sites, which struggled to get conventional payment processing because of the legal grey areas and high rates of disputed charges (chargebacks). Processing payments for "high-risk" merchants is a real and legal business, but it is not the kind of pristine origin story that later got told. By 2002, a young Austrian management consultant named **Markus Braun** had taken over as chief executive, and he would run the company for the next eighteen years.

In 2005, Wirecard pulled off a manoeuvre that quietly mattered. Rather than go through the scrutiny of a traditional initial public offering, it took over the stock-market listing of a defunct call-centre company, **InfoGenie**, in a so-called **backdoor listing** (or reverse takeover). A company that wants to be public without the full IPO process can merge into an already-listed shell, inheriting its listing. The effect was that Wirecard became a publicly traded German company with considerably less of the upfront scrutiny a fresh IPO would have brought. From the start, the company wore the credibility of a public listing without having walked fully through the front door.

From there the reported numbers climbed and climbed. Wirecard told a story investors loved: it was a high-growth, high-margin technology company riding the global shift from cash to digital payments, expanding aggressively across Asia, the Middle East, and beyond. It bought up payment businesses around the world. By 2015 it was a mid-cap darling; by September 2018 it had entered the **DAX-30**, displacing Commerzbank, an extraordinary symbolic victory. At its peak in 2018, Wirecard's market value reached roughly 24 billion euros, about \$28 billion. The chief operating officer, an Austrian named **Jan Marsalek**, ran the Asian operations and the relationships with the third-party-acquirer partners — the very part of the business that was hardest to see.

This rise is not, by itself, fraud. Companies do pivot from scrappy beginnings to respectable growth stories, and digital payments genuinely were booming. The problem was *how Wirecard made its most important business look so profitable and so cash-rich, in places almost no outside investor could see.*

### The shape of the claim: profit in Asia, cash with trustees

Here is the structure of what Wirecard told the world, and it is worth stating plainly because the whole fraud is a distortion of it. A large and growing share of Wirecard's profits — by the late 2010s, possibly half or more of the reported total — came not from its visible European processing but from its **third-party-acquirer business** run through a handful of partners in places like the Philippines, Singapore, Dubai, and elsewhere. And the cash thrown off by that business was said to be accumulating in **escrow accounts held by trustees**, eventually reported as 1.9 billion euros (about \$2.1 billion) of cash on the balance sheet.

So the two most important numbers in the entire company — its profits and its cash — both lived in the one place that was hardest for any outsider to verify: opaque partners abroad, and trustee accounts in Asia. That is the setup. The figure below traces how a number could travel from an invented merchant all the way to a celebrated balance-sheet line.

![Pipeline showing how fake third-party-acquirer revenue and escrow cash were booked into Wirecard's accounts](/imgs/blogs/wirecard-the-german-fintech-fraud-2.png)

The chain ran like this. A third-party-acquirer partner in Asia was said to process payments for a set of merchants. Those merchants generated processing fees. Wirecard booked its share of those fees as *revenue and profit*. That profit was said to accumulate as *cash in an escrow account* held by a *trustee*. And that trustee cash showed up on Wirecard's *balance sheet*, where EY's audit was supposed to confirm it. The investigations and the eventual collapse would suggest that, for the crucial Asian business, the merchants were in significant part fictional or vastly overstated, the fees were therefore largely invented, and the trustee cash at the end of the chain was a fiction propped up by forged documents. Every link looked solid; almost none of it was independently checked at the source.

## The blow-up, step by step

For years the machine ran, and the loudest voices around it were not the doubters but the regulator and the auditor — defending the company. The unravelling, when it finally came, was astonishingly fast. Here is the chronology.

### 2008: the first attack on the messenger

The pattern that would define the whole saga showed up early. In 2008, a German shareholder association published criticism of Wirecard's balance sheet. Wirecard's response set the template: rather than answer the substance, the company and its allies treated critics as the problem. Two people connected to the criticism were later prosecuted for share-price manipulation. The lesson Wirecard appeared to take from this was that attacking the critics worked.

### 2015–2016: the FT and the short sellers arrive

In 2015, the *Financial Times* began a series of articles, eventually under the banner "House of Wirecard," led by reporter **Dan McCrum**, raising detailed questions about inconsistencies in Wirecard's accounts and its acquisitions. The questions were specific and the kind a good analyst asks: why did acquired companies seem to be valued far above what their financials justified, why did the cash flows not match the reported profits, and why was so much of the growth concentrated in jurisdictions where the underlying business was impossible to inspect. In early 2016, an anonymous report by short sellers calling themselves "Zatarra" alleged fraud and money laundering at Wirecard; the stock fell sharply. Wirecard and German authorities treated the short sellers as market manipulators, and the substance of the allegations got far less official attention than the question of who was shorting the stock. The skeptics were, in effect, on trial; the company was not.

What is striking, with hindsight, is that almost every serious allegation was *checkable* in principle and *uncheckable* in practice for an outsider. McCrum could show that the numbers did not add up; he could not walk into a Manila bank and demand the account statements. The only parties who could close that gap — the auditor, with its power to confirm balances directly, and the regulator, with its power to compel disclosure — were the very parties who declined to use that power against Wirecard and instead turned it on the people raising the alarm. The journalists had the *questions* but not the *keys*; the gatekeepers had the keys but would not turn them.

### 2019: the journalists get raided, the regulator bans shorting

This is the year the establishment's defence of Wirecard reached its most extraordinary pitch. In early 2019, the *Financial Times* published a series of reports, based on internal documents and a whistleblower, alleging that senior Wirecard staff in the Singapore office had engaged in book-padding and forgery — that revenue had been fabricated and money round-tripped to make the business look bigger than it was.

The response of the German financial regulator, **BaFin** (the Federal Financial Supervisory Authority), is one of the most striking failures of supervision in modern European finance. Rather than treat the allegations as a reason to scrutinise Wirecard, BaFin **filed a criminal complaint against the FT journalists** and against short sellers for suspected market manipulation, and in February 2019 it took the rare step of imposing a **two-month ban on short-selling Wirecard shares** — a protection normally reserved for systemically important banks in a crisis, now extended to shield a single payments company from people betting it would fall. Meanwhile, German prosecutors opened an investigation into the journalists. The regulator had, in effect, picked a side, and it was not the side of the people who turned out to be right. (For how a regulator and central bank are *supposed* to think about systemic risk and market integrity, contrast this with the role of [bank regulators and the Basel framework](/blog/trading/finance/bis-and-basel-bank-regulation).)

### Late 2019: the KPMG special audit

Under mounting pressure, Wirecard's supervisory board commissioned a *special audit* from a different firm, **KPMG**, to put the FT's allegations to rest. The hope, clearly, was vindication. Instead, when KPMG reported in April 2020, it delivered something close to the opposite: KPMG said it had been **unable to verify** the bulk of the third-party-acquirer profits — that Wirecard had not provided the documents and access KPMG needed to confirm that the TPA business and its profits were real. A clean bill of health would have ended the story. An inability to verify the company's most important profits should have ended the company. The stock wobbled but did not yet break; the halo held a little longer.

### June 2020: the cash is found never to have existed

Then came the days that ended it. In June 2020, EY — finally, and far too late — pressed for direct confirmation of the 1.9 billion euros of trustee cash supposedly held in two banks in the Philippines. On **June 18, 2020**, Wirecard announced that its auditor could not confirm the existence of 1.9 billion euros (about \$2.1 billion) of cash, that the publication of its 2019 annual results would be delayed again, and that lenders might be able to call in loans. The Philippine central bank stated that the money had never entered the Philippine financial system. The two banks said to be holding the cash, **BDO** and **BPI**, said they had **no such accounts** and that documents purporting to show the balances were spurious — forgeries. The 1.9 billion euros had never existed.

On **June 22, 2020**, Wirecard admitted there was a "prevailing likelihood" that the cash balances did not exist and withdrew years of financial results. CEO **Markus Braun** resigned and was **arrested** the following day. COO **Jan Marsalek**, who had run the Asian operations at the heart of the missing money, was dismissed — and then disappeared. He fled, reportedly via a private flight, and at the time of writing remains an international fugitive, the subject of intense interest not only as a fraud suspect but over alleged ties to intelligence services. On **June 25, 2020**, Wirecard AG filed for insolvency. The stock, which had traded around 104 euros at the start of the saga's final chapter, collapsed to a few cents — a near-total wipeout for shareholders.

#### Worked example: a DAX index fund's forced wipeout

Here is a consequence most people never think about: because Wirecard was in the DAX-30, vast numbers of ordinary savers owned it without ever choosing to. Index funds and pension funds that track the DAX are *required* to hold every member of the index in proportion to its size. When Wirecard joined the DAX in 2018 at a market value of roughly 24 billion euros, every euro of money tracking the index automatically bought Wirecard.

Suppose a pension fund ran a 1 billion-euro DAX-tracking portfolio. At Wirecard's peak the company was perhaps 1% to 1.5% of the index by weight; take 1.2% as a round figure. The fund therefore held about 1,000,000,000 x 0.012 = **12 million euros of Wirecard**, not because any analyst chose it but because the rules of indexing forced the purchase. When the stock fell from about 104 euros to roughly 1 euro — a drop of about 99% — that 12 million-euro position became worth about 120,000 euros. The fund lost roughly **11.9 million euros** on a holding it never actively picked, and it could not easily sell on the way down: a stock can be suspended, can gap straight through your exit, and an index fund is structurally a buyer, not a timer.

The intuition: a fraud inside a national index is not a problem only for the gamblers who bet on it; through index and pension funds it quietly reaches into the retirement savings of people who never heard the company's name.

The before-and-after is brutal in its simplicity.

![Before-and-after comparison of Wirecard's claimed balance sheet versus the insolvent reality once the missing cash is removed](/imgs/blogs/wirecard-the-german-fintech-fraud-3.png)

Remove one line — the 1.9 billion euros of cash that did not exist — and the celebrated DAX-30 fintech was not a profitable growth company at all. It was a business with a modest real European operation, a large pile of debt, fictional Asian profits, and no cushion. The cash that everyone treated as the safest, most solid number on the books was the number that was pure fiction.

## The mechanism dissected: why it actually broke

A timeline tells you *what* happened; this section tells you *why* the structure stood for so long and why the people whose job it was to catch it did not. There were four interlocking mechanisms.

### Mechanism one: fabricated third-party-acquirer revenue

Recall the structural weakness of a TPA arrangement: the merchants, transactions, and contracts sit one company removed from the firm reporting the revenue. Wirecard appears to have exploited this gap directly. Investigations and the later insolvency administrator's work suggested that much of the third-party-acquirer business — the part claimed to generate a large share of group profits — was either grossly overstated or largely fictional. Merchants that were supposed to be generating processing volume could not be verified to exist or to be doing meaningful business. Revenue that should have come from real fees on real transactions was, to a substantial degree, an entry created to make the company look bigger and more profitable than it was.

This is the deepest layer of the fraud, and the one KPMG's special audit ran straight into: not "we found the profits and they are smaller than claimed," but "we could not verify that these profits are real at all." When the most important profits of a company cannot be independently confirmed even by a forensic team given the assignment of confirming them, that is not a presentation problem. It is the alarm.

#### Worked example: the scale of the fabricated profit

Let us put rough numbers on it to feel the scale. In its 2018 reporting, Wirecard claimed group revenue of roughly 2 billion euros and operating profit (EBITDA) of roughly 560 million euros. By various estimates from the investigations, the third-party-acquirer business based in Asia was claimed to account for *roughly half* of the company's profits in its later years.

Suppose, conservatively, that \$250 million of annual operating profit was attributed to the TPA business and that the bulk of it was fictional. Over the four years from 2015 to 2019, that is on the order of \$1 billion of *profit that was reported but never economically earned*. To a market valuing the company on its earnings and growth, every one of those fictional dollars was multiplied. If investors were paying, say, 30 times earnings for a fast-growing fintech — meaning they valued each \$1 of annual profit at about \$30 of market value — then \$250 million of fake annual profit could have been supporting roughly 250 million x 30 = \$7.5 billion of market capitalisation built on nothing. The arithmetic is approximate, but the intuition is exact.

The intuition: in a company valued on its earnings, a dollar of fabricated profit does not cost the fraud one dollar of credibility — it manufactures tens of dollars of market value, which is why the incentive to invent profit is so corrosive.

### Mechanism two: the escrow-balance fiction

Fabricated profit creates a second problem for a fraudster: where is the cash? If you book a profit, an honest balance sheet expects the corresponding cash to show up somewhere. Wirecard's answer was the escrow story — the profits were accumulating as cash held by trustees in Asia. This neatly explained why the money was not in Wirecard's own German accounts where it could be easily seen, while still letting the company report a large, comforting cash balance. The reported 1.9 billion euros of trustee cash was the *plug* that made the fabricated-profit story balance.

The figure below shows how that cash sat in the structure — two intermediaries removed from Wirecard, behind partners and trustees that no one independently confirmed at the source.

![Tree diagram of the third-party-acquirer and trustee structure holding Wirecard's reported cash](/imgs/blogs/wirecard-the-german-fintech-fraud-7.png)

To make the escrow story hold, the balances had to be *confirmed* to auditors. This is where the forged bank documents come in. When confirmation was sought, documents appeared purporting to show the balances at the named banks. Those documents — including, in the end, letters supposedly from BDO and BPI in the Philippines — were spurious. The whole point of an escrow-and-trustee structure is to create an air of independent safekeeping; here it was used in reverse, as a stage on which a fictional cash balance could be made to look custodied and confirmed.

#### Worked example: the escrow balance as a share of total assets

Consider how central that single fictional line was. In its last full reported balance sheet, Wirecard claimed total assets of roughly 5.8 billion euros. The trustee cash in question was 1.9 billion euros. That is 1.9 / 5.8 = about **33% of all reported assets**, concentrated in one line, sitting in accounts in the Philippines, behind third parties, confirmed only by documents that turned out to be forged.

Now do the removal. Take 1.9 billion euros out of 5.8 billion and you are left with about 3.9 billion euros of assets. Against that, Wirecard had on the order of 3.5 billion euros of financial debt and other liabilities, plus the reality that a chunk of the *remaining* assets — goodwill from acquisitions and receivables from the same dubious partners — was itself of doubtful value. The equity cushion that investors believed protected them was not thin; once the phantom cash was removed, it was gone, and the company was insolvent.

It helps to see where that fictional cash sat in the asset stack, because its position is part of why it was so dangerous. Investors rank assets by how solid they are: cash is the gold standard, the asset you can spend tomorrow with no questions; receivables (money others owe you) are softer, because the debtor might not pay; goodwill — the premium paid above tangible value in past acquisitions — is the softest of all, because it is an accounting estimate, not a thing. The figure below stacks Wirecard's assets from solid to soft.

![Stack diagram showing the missing trustee cash as the top layer of Wirecard's reported asset stack](/imgs/blogs/wirecard-the-german-fintech-fraud-6.png)

The cruel irony is that the *least* solid asset of all — a phantom cash balance backed by forged letters — was reported as the *most* solid layer, the cash at the top of the stack. The number investors trusted most, the one they would have stress-tested last, was the one that was pure invention. When it vanished, what remained was a thin layer of real European processing under a heap of soft goodwill and disputed receivables. The whole pyramid had been resting on its phantom apex.

The intuition: when a third of a company's assets sits in a single, hard-to-verify, far-away line that is reported as its safest asset, the entire investment case rests on whether that one number is real — and here it was not.

### Mechanism three: an auditor that never confirmed the cash at the source

Now the failure that should never have happened. The single most basic safeguard against exactly this fraud is the *bank confirmation*: the auditor asks the bank *directly* how much money is in the account, and the bank replies *directly* to the auditor. For roughly a decade, EY signed clean audit opinions on Wirecard while the largest and most important cash balances — by 2019, the 1.9 billion euros in trustee accounts — were not confirmed by that direct, independent route. Instead, the audit appears to have leaned on documents and confirmations that flowed through the company and the trustees rather than coming straight from the banks themselves. When EY finally did insist on direct confirmation from BDO and BPI in 2020, the banks said the accounts did not exist, and the fraud collapsed within days.

The point is not that confirming cash is hard. It is that confirming cash is *easy and fundamental* — which is what makes the failure so damning. A first-year auditor is taught that you do not accept a cash balance on the strength of paperwork handed to you by the client; you go to the bank. For the most material assets of a DAX-30 company, that elementary step, properly done at the source, would have ended the fraud years earlier. EY has faced regulatory penalties and lawsuits over the audits. The graph below lays out the web of gatekeepers and how each one ended up pointing away from the hole rather than at it.

![Graph showing the web of Wirecard, its third-party-acquirer partners, escrow trustees, the auditor EY, and the regulator BaFin around the missing cash](/imgs/blogs/wirecard-the-german-fintech-fraud-4.png)

#### Worked example: the audit fee versus the check that was not done

Here is the economics of the missed check, and it is uncomfortable. Over the years it audited Wirecard, EY earned substantial fees — running into the millions of euros annually in the later years, and a meaningful sum across the decade in total. Set that against the cost of the single procedure that would have caught the fraud.

A direct bank confirmation for a cash balance is, in mechanical terms, a request letter sent straight to the bank and a reply sent straight back to the auditor. For a 1.9 billion-euro balance, the marginal cost of insisting on doing it properly — going to BDO and BPI directly, refusing to accept intermediated paperwork, and declining to sign until the banks themselves confirmed — was effectively the cost of a few letters and the institutional willingness to delay an opinion. Call it, generously, a few thousand euros of effort and the discomfort of pushing a powerful client.

So the trade was roughly this: millions of euros of fees collected, against a few thousand euros of work and some awkwardness, to confirm a balance equal to *a third of the client's assets*. The fraud survived in the gap between what the auditor was paid and the elementary check it did not force through at the source.

The intuition: the cheapest, most basic audit procedure in existence — ask the bank directly — was the one that mattered most, and it was the one that, for the crucial balances, did not get done properly until the very end.

### Mechanism four: a captured and defensive regulator

The final mechanism is the one that makes Wirecard distinctively a *German* scandal rather than a generic one. In most frauds the regulator is, at worst, asleep. In Wirecard's case the regulator was *active* — but pointed in the wrong direction. BaFin investigated the journalists and short sellers, filed criminal complaints against the people raising the alarm, and banned short-selling of the stock to protect it from skeptics. Several factors fed this: a national pride in Wirecard as a home-grown technology champion; a regulatory structure in which it was unclear whether Wirecard was a "technology company" or a financial institution, leaving its supervision fragmented; and a reflex to treat foreign journalists and short sellers betting against a German DAX champion as the threat, rather than as a warning. The matrix below lines up these red flags against a famous predecessor, because the pattern is not new.

![Matrix comparing Wirecard's red flags against the parallel red flags in the Enron fraud](/imgs/blogs/wirecard-the-german-fintech-fraud-5.png)

When the regulator that is supposed to protect investors instead deploys its powers against the people warning investors, it does more than fail to catch the fraud. It actively prolongs it, by lending the company the state's own credibility and by raising the cost — legal, financial, reputational — of telling the truth. The short sellers and the FT were not just ignored; for a time they were treated as the criminals.

It is worth understanding *why* a regulator would behave this way, because the answer is not simple corruption. Several ordinary forces combined into an extraordinary failure. First, **jurisdictional confusion**: Wirecard had a banking subsidiary (Wirecard Bank) that BaFin supervised, but the parent holding company presented itself as a *technology* firm, and technology firms are not prudentially supervised the way banks are. The most important entity — the one reporting the fictional Asian profits and the phantom cash — fell into a supervisory gap where no single regulator felt clearly responsible for confirming the substance of the accounts. Second, **national pride**: Wirecard was Germany's rare home-grown technology champion in a country whose industrial giants are mostly old. The instinct to defend it against foreign critics, especially Anglo-American journalists and short sellers, was strong and politically resonant. Third, a **deep-seated suspicion of short selling** in continental Europe, where betting against a company is often viewed less as legitimate skepticism than as a destructive, manipulative act. Stack these together and a regulator can convince itself, in good faith, that the real threat is the people attacking a national champion rather than the champion itself.

That is precisely what makes the failure instructive. You do not need a corrupt regulator to get a captured outcome; you need a regulator whose structure, incentives, and cultural instincts all point it at the wrong target. The reform that followed — restructuring BaFin, clarifying its authority over conglomerates, and replacing its leadership — was an attempt to correct exactly these structural defects, not to root out venality that was never the main problem.

## The aftermath: arrests, reforms, and a fugitive

The collapse triggered consequences across several fronts.

**The people.** Markus Braun was arrested in June 2020 and went on trial in Munich beginning in December 2022, charged with fraud, breach of trust, and accounting manipulation; he has maintained that he too was a victim of others' deceit. A key witness has been **Oliver Bellenhaus**, a former senior Wirecard manager who turned state's witness and described the business as built on fabrication. **Jan Marsalek**, the COO who ran the Asian operations, vanished in June 2020 and remains a fugitive; reporting has since linked him to Russian intelligence, turning a fraud case into an espionage story whose full shape is still emerging.

**The auditor.** EY came under intense scrutiny from regulators and faced lawsuits from burned investors. Germany's audit oversight body sanctioned EY and individual auditors, and the affair became a case study in how a Big Four audit can fail at the most basic task of confirming cash.

**The regulator.** BaFin's handling — investigating the journalists, banning short-selling, missing the fraud — was widely condemned. Its president was replaced, the institution was restructured and given clearer authority over conglomerates like Wirecard, and the episode forced a broader German reckoning with how a national champion had been shielded rather than scrutinised. The German parliament held an inquiry that pulled in regulators, the finance ministry, and the auditors.

**The journalists, vindicated.** Dan McCrum and the *Financial Times*, investigated and pressured for years, were comprehensively vindicated. The episode became a landmark in financial journalism — a reminder that the people the establishment treats as troublemakers are sometimes the only ones doing the establishment's job. (The dynamic of skeptics being proven right after being attacked echoes through cases from [Enron's short sellers](/blog/trading/finance/enron-2001-accounting-fraud) to [Madoff's doubters](/blog/trading/finance/madoff-ponzi-scheme).)

#### Worked example: a short seller's payoff as the stock collapses

Consider a short seller who, like several real ones, bet against Wirecard near its peak. Suppose they shorted the stock at about 104 euros per share — borrowing shares and selling them at that price — across a position of, say, 100,000 shares. The proceeds from the short sale are 100,000 x 104 = 10.4 million euros, held as the obligation to buy the shares back later.

When the fraud broke and the stock collapsed to roughly 1 euro, the short seller buys back 100,000 shares for about 100,000 x 1 = 100,000 euros and returns them, closing the position. The profit is the difference: 10.4 million - 0.1 million = **about 10.3 million euros** on that position, a gain of nearly 99% of the amount initially sold short.

But notice the asymmetry that makes short selling brutal, and that explains why so few stick with it. In the years *before* the collapse, the same short seller watched the stock climb from the teens into the hundreds while BaFin banned shorting and prosecutors investigated short sellers as manipulators. A short position that is "right" but early bleeds money and legal risk the whole way up; if the trader were forced to close out — by margin calls, by the short-selling ban, or by sheer exhaustion — before June 2020, the eventual collapse would have paid them nothing. Being correct about a fraud is not enough; you must also survive long enough, financially and legally, to be paid for it.

The intuition: the short seller's eventual windfall, from about 104 euros to near zero, was the mirror image of every other shareholder's wipeout — but earning it required surviving years of being right too soon and treated as the wrongdoer.

## Common misconceptions

A few beliefs about Wirecard are widespread and worth correcting directly.

**"Wirecard was a fake company with no real business."** Not quite, and the distinction matters. Wirecard had a genuine, sizeable payment-processing operation in Europe — real merchants, real transactions, real fees. The fraud was concentrated in the *third-party-acquirer* business abroad and the trustee cash it supposedly generated. The danger of a fraud wrapped around a real core is precisely that the real part lends credibility to the fictional part. A company that is entirely fake is easier to spot than one that is two-thirds real and one-third invented in the place no one can see.

**"The auditors just got fooled by sophisticated forgeries."** The forged bank letters were real, but the deeper failure was that the most basic safeguard — confirming cash *directly with the bank* rather than through intermediated documents — was not properly applied to the most material balances for years. A forgery only works if no one performs the independent check that would expose it. The lesson is not "the fraud was too clever to catch"; it is that the elementary check was not forced through until the very end.

**"This was a problem of bad luck or a few rogue employees."** The structure of the fraud — fabricated profits in opaque jurisdictions, a plug of phantom trustee cash, intermediated confirmations — required design and persistence over years, and it survived because multiple gatekeepers failed at once: the auditor on confirming cash, the regulator on scrutinising the company, and a market on asking where the cash actually was. Systemic failures, not bad luck.

**"BaFin was simply asleep at the wheel."** Worse than asleep. BaFin was awake and active, but it deployed its powers against the journalists and short sellers raising the alarm and banned short-selling to protect the stock. A regulator that defends the accused and prosecutes the accusers is not merely negligent; it prolongs the fraud and amplifies its damage.

**"Short sellers caused the collapse by attacking the company."** Short sellers profit from a collapse, but they did not cause Wirecard's. The cash either existed or it did not, and it did not. The short sellers and the FT *identified* a fraud that was already there; they were the diagnosis, not the disease. Blaming them is like blaming the smoke detector for the fire.

**"A DAX listing and a Big Four audit mean the numbers are safe."** This is the misconception the whole case exists to puncture. A prestigious listing, a Big Four auditor, and a national regulator are a chain of gatekeepers, and a chain provides protection only if at least one link actually does the checking. Here, for the number that mattered most, none did — until it was far too late.

## How it echoes in other markets

Wirecard is not a freak event. It sits in a long line of frauds that share a recognisable skeleton: profits that do not turn into verifiable cash, gatekeepers that fail, and skeptics that are punished before they are vindicated.

**Enron (2001).** The clearest American parallel. Enron booked profits its cash never matched, hid the truth in off-balance-sheet entities no one scrutinised, and saw its conflicted auditor sign off until short sellers and a journalist pulled the thread. Wirecard repeated the pattern a continent and two decades away. The detailed [Enron walkthrough](/blog/trading/finance/enron-2001-accounting-fraud) is the closest companion to this story.

**Parmalat (2003).** Often called "Europe's Enron," the Italian dairy giant Parmalat collapsed when a supposed 3.95 billion-euro account at Bank of America turned out not to exist — confirmed by a document that was forged. The resemblance to Wirecard's phantom Philippine cash is almost exact: a giant, comforting cash balance, in a faraway bank, that simply was not there. The same elementary check — ask the bank directly — would have caught both.

**Satyam (2009).** India's "Satyam scandal" saw the IT-services company's founder admit that more than \$1 billion of cash on its balance sheet was fictitious, inflated over years to match fabricated revenue. Again: fabricated profit, a phantom cash plug, an audit that did not confirm the balance.

**Luckin Coffee (2020).** In the same year Wirecard collapsed, the Chinese coffee chain Luckin disclosed that hundreds of millions of dollars of its sales had been fabricated. A short seller's report (from Muddy Waters) had alleged exactly that before the company admitted it — once more, the skeptic was right and early.

**Bernie Madoff (2008).** A different mechanism — a Ponzi scheme rather than fabricated operating profit — but the same fatal question. Madoff reported steady returns and account balances that no one independently verified at the source, and the regulator (the SEC) ignored a whistleblower, Harry Markopolos, who had done the arithmetic and shown the returns were impossible. The [Madoff case](/blog/trading/finance/madoff-ponzi-scheme) is the purest example of "no one checked whether the assets were actually there."

The common thread across all of them: *reported profit is an opinion and reported cash is a claim until someone independent confirms the money is real.* Every one of these frauds died the moment that confirmation was finally attempted. And in nearly every one, a skeptic had been shouting the answer for years.

## When this matters to you

You are unlikely to run a forensic audit, but the Wirecard pattern shows up at every scale, and a few habits of mind travel well.

**Follow the cash, not the profit.** Whenever you assess a company — as an investor, an employee deciding whether to take stock, or a lender — the question that cuts deepest is whether reported profits turn into real, verifiable cash, and where that cash actually sits. A company reporting big profits while its cash piles up in hard-to-see places, far from where the business operates, deserves harder questions, not a higher valuation.

**Treat the credibility halo as a hypothesis, not a guarantee.** "It is listed, audited, and regulated" describes a chain of gatekeepers, each of which can fail, and which can all fail together. The presence of prestigious names is a reason to expect checking happened, not proof that it did. Wirecard had every name on its side.

**Notice who is attacking the skeptics.** When a company, a regulator, or a national press treats the people asking hard financial questions as the villains — investigating journalists, banning short sellers, smearing critics — that reaction is itself a red flag. Honest companies answer the substance. The Wirecard, Enron, and Luckin skeptics were all attacked before they were proven right.

**Independence is the whole point of a check.** The reason a bank confirmation must go *directly* between auditor and bank, and the reason an escrow trustee must be genuinely neutral, is that a check routed through the party being checked is not a check at all. Whenever you are relying on a confirmation, ask who actually produced it and whether they had any reason to lie.

### Further reading and where to go next

To see the same skeleton in its most studied form, read the [Enron accounting-fraud walkthrough](/blog/trading/finance/enron-2001-accounting-fraud) and the [Madoff Ponzi scheme](/blog/trading/finance/madoff-ponzi-scheme); together they cover the two great families of "the assets were not there" fraud. To understand the legitimate, thriving version of the business Wirecard pretended to dominate, see how [the real fintech disruptors built payment processing](/blog/trading/finance/fintech-disruptors-stripe-paypal-ant). And to understand the institutions whose gatekeeping failed — auditors, regulators, and the bankers who lent to and underwrote Wirecard — the field guide to [how an investment bank makes money](/blog/trading/finance/inside-an-investment-bank-how-they-make-money) shows where the incentives that surround a company like this one come from.

Wirecard's epitaph is not complicated. A company can be celebrated, indexed, audited, and regulated, and still be missing 1.9 billion euros — because the celebration, the index, the audit, and the regulator are only as good as whether anyone bothered to ask the bank if the money was actually there. For years, no one with the power to matter did. When they finally did, the answer took about a week to end a company that had taken twenty years to build.
