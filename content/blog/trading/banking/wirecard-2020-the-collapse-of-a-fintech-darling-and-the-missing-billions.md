---
title: "Wirecard 2020: The Collapse of a Fintech Darling and the Missing Billions"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "How a German payments company briefly worth more than Deutsche Bank turned out to be missing 1.9 billion euros of cash that never existed, and what its failed auditors, regulator, and board teach about the gatekeepers we rely on."
tags: ["banking", "wirecard", "accounting-fraud", "audit-failure", "fintech", "payments", "corporate-governance", "regulation", "gatekeepers", "case-study"]
category: "trading"
subcategory: "Banking"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — Wirecard, a German payments company once worth more than Deutsche Bank, reported €1.9bn of cash sitting in Asian escrow accounts that simply did not exist; when its auditor finally tried to confirm the balances in June 2020, the money was not there and the company collapsed within a week.
>
> - The fraud was not subtle math — it was missing cash. The single fastest check in all of auditing, writing to the bank that supposedly holds the money, is the one that was never properly done for years.
> - Three layers of protection failed at once: the external **auditor** (EY) signed clean accounts without confirming the cash; the **regulator** (BaFin) defended the company, banned short selling, and investigated the journalists who were right; the **board** trusted management.
> - The warnings were public and specific from 2015 onward. Being early and being right are not the same thing as being believed — short sellers and the *Financial Times* were dismissed as a market attack.
> - The one number to remember: **€1.9bn** — roughly all of Wirecard's reported cash and many times its reported annual profit. The hole was not a rounding error. It was the company.

In September 2018, Wirecard joined Germany's DAX 30, the index of the country's largest listed companies. It replaced Commerzbank, one of the country's oldest banks. For a brief stretch the market valued this payments-technology firm from a Munich suburb at around €24bn — more than Deutsche Bank, the institution that had financed German industry for a century and a half. Wirecard was the proof, finance ministers and tech boosters said, that Europe could produce a payments champion to rival the Americans. Analysts had price targets that implied it would keep doubling. Index funds had to own it because it was in the index. It was, in every sense the word is used by people selling something, a *darling*.

Less than two years later, on 25 June 2020, Wirecard filed for insolvency. A week earlier its auditor had refused to sign the accounts because €1.9bn of cash that the company claimed sat in trust accounts in the Philippines could not be found. The two banks that were supposed to hold it said they had never had any relationship with Wirecard at all. The chief executive was arrested. The chief operating officer fled and remains a fugitive. The €24bn of market value went to roughly zero.

This post is about how that happened, and why it is one of the most important banking-and-finance failures of the modern era even though Wirecard was not, in the everyday sense, a bank failure. It was a payments company that *owned* a bank, and the lesson it teaches is not about leverage or a deposit run. It is about the **gatekeepers** — the auditors, regulators, and boards we trust to verify that a company is what it says it is — and what happens when all of them fail at the same time. The figure below is the mental model for the whole story: years of public warnings, then a collapse that took only days once someone finally checked whether the cash was real.

![Wirecard timeline from first questions in 2015 to insolvency in June 2020](/imgs/blogs/wirecard-2020-the-collapse-of-a-fintech-darling-and-the-missing-billions-1.png)

## Foundations: payments, escrow, auditors, and the people meant to catch fraud

Before we can see what went wrong, we need the vocabulary. This story lives at the intersection of how electronic payments work and how companies are checked, so we will build both from zero. A practitioner can skim; if any term here is new to you, do not skip it, because the entire fraud hides in the gap between what these words *mean* and what Wirecard *claimed*.

### What a payment processor and an acquirer actually do

When you tap a card at a shop, your money does not teleport to the merchant. A chain of intermediaries moves it, and each takes a sliver. Two roles matter here.

A **payment processor** is the company that handles the technical traffic of a card transaction — taking the request from the till, routing it to the card networks (Visa, Mastercard), getting an approval or a decline back, and recording it. It is the switchboard operator for money.

An **acquirer** (or *acquiring bank*) is the institution that holds the merchant's account and is contractually on the hook to the card networks for that merchant's transactions. The acquirer *acquires* the merchant's business: it signs them up, takes on the risk that the merchant is a fraud or goes bust mid-refund, and in return collects a cut of every sale. That cut is the **merchant discount rate** — the few percent a shop pays on each card sale. (We unpack exactly how that fee is split among the issuer, the acquirer, and the network in the companion post on [the cards business](/blog/trading/banking/the-cards-business-issuing-acquiring-interchange-and-the-mdr-split).)

Wirecard did both. It processed payments and it was an acquirer, and it owned a licensed German bank, **Wirecard Bank AG**, that gave it the regulatory standing to settle money. That bank ownership matters: it is why Germany's financial regulator was involved at all, and why people assumed Wirecard was more closely watched than a pure tech firm.

### Third-party acquiring (TPA) — the part that turned out to be fiction

Here is the crucial wrinkle. A company like Wirecard cannot have its own license and presence in every country on earth. So in places where it had no local license — much of Asia, the Middle East — it claimed to use **third-party acquirers (TPA)**: local partner firms that did the actual acquiring on Wirecard's behalf, with Wirecard taking a referral cut of the revenue.

On paper, TPA is a normal arrangement. In practice, at Wirecard, three TPA partners — in Dubai, the Philippines, and Singapore — were reported to generate roughly **half of the company's revenue and the great majority of its profit** by 2018. These partners were small, opaque, and hard to verify. The money they supposedly earned on Wirecard's behalf was said to be held, on Wirecard's behalf, in escrow.

That is the fiction. The real third-party-acquiring flow and the fabricated escrow story sit side by side in the next figure: a genuine payment ends in settled cash that a bank can confirm; Wirecard's version ended in a claim about cash that no bank would confirm.

![Real third party acquiring flow compared with the fabricated escrow story](/imgs/blogs/wirecard-2020-the-collapse-of-a-fintech-darling-and-the-missing-billions-2.png)

### What an escrow account is

An **escrow account** is money held by a neutral third party (a *trustee*) on behalf of two other parties, released only under agreed conditions. When you buy a house, the deposit often sits in escrow with a lawyer until the deal closes. The point of escrow is trust: the money is real, it is segregated, and a named third party vouches for it.

Wirecard told the world that the cash its Asian partners earned was parked in escrow accounts at banks in the Philippines, controlled by a trustee, on Wirecard's behalf. By the end of 2019 that escrow cash was reported at **€1.9bn**. It was the single largest asset on the balance sheet, and it was the proof that all those profits were real — because, the company said, you could see the money.

### What an external audit and a bank confirmation are

A company's accounts are prepared by its own management. Because management has every incentive to flatter the numbers, the law requires an independent **external auditor** — a separate accounting firm — to examine the books and issue an opinion on whether the financial statements give a true and fair view. The big firms doing this are EY, Deloitte, KPMG, and PwC, the so-called Big Four. Wirecard's auditor was **EY** (Ernst & Young), which signed Wirecard's accounts for about a decade.

The single most basic procedure in any audit is **confirming cash**. If a company says it has money in a bank account, the auditor does not just look at a bank statement the company hands over — because the company could have forged it. The auditor sends an independent request, called a **bank confirmation**, *directly to the bank*, and the bank replies *directly to the auditor*, stating the exact balance on a given date. It is the audit equivalent of phoning the landlord yourself instead of trusting the tenant's word that the rent is paid. It is cheap, it is routine, and it is the first thing you do when an asset is "cash".

Hold onto that, because the entire Wirecard fraud comes down to this: for the €1.9bn that mattered most, that direct confirmation was, for years, not properly obtained.

There is a second concept buried in the word "audit" that is worth pulling out, because it is where so much public confusion lives. An audit is **not a fraud investigation**. The auditor's job, under the standards as they are written, is to obtain *reasonable assurance* that the accounts are free of material misstatement — not to act as a detective assuming the management are criminals. That sounds like a get-out clause, and auditors have leaned on it. But "reasonable assurance" still has a floor, and that floor for cash is non-negotiable: you confirm it with the bank. The defense "we are not detectives" can excuse a failure to unravel a clever scheme of related-party transactions. It cannot excuse not phoning the bank about €1.9bn. The distinction matters because it is exactly the line EY tried to walk after the collapse, and exactly the line the standards do not let it walk.

### Why a balance sheet has to balance — and why that traps a fraudster

One more building block, because the whole mechanism of the fraud is a consequence of it. A **balance sheet** has two sides that must be equal: on one side the company's assets (what it owns — cash, loans, buildings); on the other side its liabilities plus equity (what it owes, plus the owners' stake). The two sides are equal by construction — that is why it is called a *balance* sheet. The companion post on [reading a bank's balance sheet](/blog/trading/banking/reading-a-bank-balance-sheet-assets-liabilities-and-equity) builds this from scratch; here we only need one consequence.

The consequence is this: **you cannot invent profit without inventing a matching asset.** If your income statement claims you earned an extra €200m this year, that €200m has to show up somewhere on the asset side of the balance sheet — as cash collected, or a receivable owed to you, or some investment. Real profit becomes a real asset automatically. Fake profit has nowhere to land, so the fraudster must conjure a matching fake asset out of thin air or the books will not balance and even a careless auditor will notice. Wirecard's fake asset of choice was cash — the escrow balances — because cash is the most reassuring thing on any balance sheet. The cruel irony is that cash is also the easiest asset to verify, so by choosing the most convincing fake asset, the fraudsters chose the one with the simplest possible check standing between them and exposure. That check just was not made.

### Short sellers, and why they are gatekeepers too

A **short seller** is an investor who profits when a stock *falls*. They borrow shares, sell them, and hope to buy them back later at a lower price, pocketing the difference. Short sellers are widely disliked — they are betting against a company, sometimes loudly — but they serve a function the rest of the market does not: they have a financial incentive to *find fraud*. A normal analyst earns nothing by discovering that a company is rotten. A short seller earns a fortune. That makes short sellers, alongside investigative journalists, an informal layer of the gatekeeping system. At Wirecard, they were right for years and treated as the criminals.

### Accounting fraud, and the gatekeeper system

**Accounting fraud** is the deliberate misstatement of a company's financial records to deceive investors, lenders, or regulators — recording revenue that did not happen, hiding losses, or, as here, inventing assets that do not exist. It is distinct from mere error: fraud is on purpose.

The system that is supposed to catch it has several layers, each independent:

- the company's own **board and audit committee**, who oversee management;
- the **external auditor**, who verifies the accounts;
- the **regulator**, who supervises markets and, for a bank, its safety and soundness;
- and the **market itself** — analysts, journalists, and short sellers who probe.

The grim insight of Wirecard is that **fraud of this scale needs every one of those layers to fail at once**, and at Wirecard they did. The rest of this post walks through each.

This connects to the spine of this whole series. A bank is a confidence machine: it works only as long as people trust that the assets backing the deposits and the equity are real. Wirecard was not a maturity-transformation failure — it did not borrow short and lend long and get caught by a rate move. But it was the purest possible failure of the *confidence* the entire financial system runs on, because the thing everyone trusted to be verified — the cash — never existed, and the people paid to verify it did not.

## The business, and the third-party-acquiring fiction

To understand the fraud you have to understand what was real and what was not, because the two were braided together. This is what makes a good fraud hard to catch: it is mostly true.

Wirecard's core European business was, by most accounts, a genuine payments operation. It processed real transactions for real merchants, owned a real bank, and issued real prepaid cards. That business made money, though never as much as the headline numbers suggested.

The problem was the growth story. A payments processor in a competitive European market grows at maybe the rate of card spending — high single digits, low teens in a good year. That does not justify a valuation richer than a major bank. To be a *darling*, Wirecard needed to be growing at 30 percent or more, with fat margins, in exciting new markets. The third-party-acquiring business in Asia and the Middle East provided exactly that narrative: enormous, fast-growing, high-margin revenue from places too far away and too opaque for anyone to easily check.

#### Worked example: how fake revenue inflates a valuation

Let us make the incentive concrete with round numbers, because this is the engine of the whole thing.

Suppose a real payments business earns €500m of revenue and €100m of profit, growing 10% a year. A market that pays, say, 20 times earnings for steady growth values it at €100m × 20 = **€2bn**.

Now bolt on fake third-party-acquiring revenue. Say you invent €500m of extra revenue at a juicy 40% margin, so €200m of extra "profit", and you tell the market this new business is growing 35% a year. Total reported profit is now €300m. But fast growth earns a richer multiple — say 35 times earnings. The market now values you at €300m × 35 = **€10.5bn**.

You added €200m of fictional profit and the share price rewarded you with roughly €8.5bn of extra market value — over forty times the fake profit. That is the leverage of a growth multiple: a euro of *invented* earnings, dressed as fast-growing, is worth far more in market value than a euro of real, slow earnings.

The one-sentence intuition: **fraudulent revenue does not just add to the valuation, it re-rates the whole company to a higher multiple — which is precisely why the fraud has to keep growing or the story dies.** Wirecard's reported numbers had to keep accelerating, and so the fictional cash pile had to keep growing too.

But invented profit creates a problem. If your income statement says you earned €200m extra last year, your balance sheet has to show where that €200m *went*. Real profit turns into real cash or real assets. Fake profit has nowhere to go — unless you invent a matching asset. That matching asset was the escrow cash. Every euro of fictional profit needed a euro of fictional cash sitting somewhere to balance the books. By 2019 that "somewhere" was €1.9bn of escrow in the Philippines.

Why were the third-party acquirers so hard to check? Three reasons, each deliberate. First, **geography**: the partners were in Dubai, the Philippines, and Singapore, far from the German auditors and analysts, in jurisdictions where local-language documents and unfamiliar banks gave plausible cover. Second, **opacity**: these were small private firms, not listed companies with their own audited accounts, so there was no independent paper trail to cross-check against. Third, **the trustee structure**: the cash was not held in Wirecard's own name in a Wirecard-controlled account that an auditor could log into. It was held by an intermediary trustee "on behalf of" Wirecard, which inserted a layer of paperwork — the trustee's letters and statements — between the auditor and the bank. Every one of these features added a place for a document to be forged and a reason for a confirmation to be delayed.

There is also a darker mechanic the *FT*'s reporting pointed to, called **round-tripping**. The suspicion was that money was being moved in circles — out of one Wirecard entity, through partners and intermediaries, and back in again — so that the same euros could be counted as fresh "revenue" on each lap. If true, this means some of the cash flows were real in the narrow sense that money did move, but the *revenue* they were dressed up as was fictional, because no genuine outside customer was paying it. Round-tripping is one of the oldest tricks in accounting fraud precisely because it produces real-looking bank movements that can fool an auditor who follows the cash a short distance but never asks where it ultimately came from. The lesson for the reader is that "we saw money move" is not the same as "a real customer paid us" — and verifying the difference requires confirming with the *source* of the funds, not just observing a transfer.

So the picture by 2019 was a company with a genuine but modest European payments business, wrapped in a fictional Asian business that supplied the growth, the margins, and the headline profits — and a €1.9bn cash pile that existed only as paperwork to make the fiction balance.

## The missing €1.9bn

So we arrive at the number. By the 2019 accounts, Wirecard reported that roughly **€1.9bn** of its cash was held in escrow accounts in the Philippines, controlled by a trustee, representing the accumulated earnings of its Asian third-party-acquiring partners.

To see how absurd this was, you have to put it next to the rest of the company's own reported figures.

![Missing 1.9 billion euros compared with Wirecard reported cash equity and profit](/imgs/blogs/wirecard-2020-the-collapse-of-a-fintech-darling-and-the-missing-billions-5.png)

#### Worked example: the €1.9bn as a share of the balance sheet

Wirecard's 2018 accounts reported total cash and cash equivalents of roughly €3.2bn (about \$3.5bn at 2020 exchange rates). The escrow portion was about €1.9bn (about \$2.1bn) of that. So the share of the company's cash that was a fiction was:

€1.9bn ÷ €3.2bn ≈ **60%**.

Now compare it to profit. Wirecard reported net profit of about €350m (about \$390m) for 2018. The missing cash was therefore:

€1,900m ÷ €350m ≈ **5.4 times** a full year's reported profit.

And against equity — the shareholders' stake, reported at roughly €2.0bn — the missing cash was nearly the *entire* equity of the company:

€1.9bn ÷ €2.0bn ≈ **95%**.

The one-sentence intuition: **the hole was not a corner of the balance sheet; it was most of the cash, several years of profit, and nearly all the equity — which is why, the moment it was confirmed missing, there was no company left.** A bank can survive a loss that is small relative to its capital. A loss equal to almost all your equity is not a loss; it is the end.

This is also why the fraud was, in a strange way, simple. It was not a complex web of derivatives or off-balance-sheet vehicles that a smart accountant might be fooled by. It was *cash*. The company said it had €1.9bn in two bank accounts. Either the money was there or it was not. There is no clever interpretation. The whole edifice rested on the most checkable claim a company can make — and on nobody checking it properly.

## The FT and the short sellers: the warnings nobody wanted

The most uncomfortable part of the Wirecard story is that it was not uncovered by the auditors or the regulator. It was uncovered, in public, in real time, by journalists and short sellers — and they were attacked for it for years.

The *Financial Times* began publishing critical pieces under the banner "House of Wirecard" in 2015, raising questions about the company's accounting and its acquisitions. Short sellers had circulated reports questioning the Asian business even earlier. In 2016, an anonymous report under the name "Zatarra" alleged fraud and money laundering; Wirecard called it market manipulation, and German authorities investigated the short sellers behind it rather than the allegations.

The reporting sharpened. In early 2019, the *FT*'s Dan McCrum published a series based on documents from a whistleblower inside Wirecard's Singapore office, describing what looked like a scheme to inflate revenue and profits through fake transactions and round-tripping of cash. In October 2019 the *FT* published documents suggesting that the bulk of Wirecard's reported profits came from those three opaque third-party partners — and that the underlying business might be far smaller than claimed.

The pattern of warnings and the responses they drew is the single most damning summary of the whole affair.

![Matrix of Wirecard warnings and the responses they drew](/imgs/blogs/wirecard-2020-the-collapse-of-a-fintech-darling-and-the-missing-billions-7.png)

Each warning was specific, evidenced, and broadly correct. Each was met not with an independent check of the cash, but with a defense of the company and an attack on the messenger. Wirecard sued the *FT*. It hired investigators to surveil journalists and critics. And the regulator, as we will see, did something almost without precedent: it took the company's side.

It is worth dwelling on why the warnings were so easy to dismiss, because the dynamic recurs in every fraud. The first defense is always **"they have a motive"**: short sellers profit if the shares fall, so their claims can be waved away as talking their book. This is true but irrelevant — a self-interested accusation can still be correct, and the way to test it is to check the facts, not to impugn the accuser. The second defense is **national and institutional pride**: the idea that a DAX-listed, EY-audited, BaFin-supervised German champion could be a fraud was, for many German investors and officials, almost unthinkable, so they reached for the explanation that flattered them — that the company was the victim of an Anglo-American attack. The third defense is **complexity as camouflage**: Wirecard's defenders could always point to the genuine European business and say, look, it is real, ignoring that the question was never whether *any* of it was real but whether the *Asian* part was. Each of these defenses substituted a story about the accusers for an examination of the accusation. The single act that would have settled it — confirming the cash with the banks — was the one nobody in authority forced until the very end.

There is a human cost worth naming. The journalists were placed under surveillance; their phones and movements were reportedly monitored by agents working for the company. The whistleblowers inside Wirecard's Asian operations took real personal risk. Being right, in this story, was for years a liability rather than a vindication — which is exactly the environment in which a fraud thrives, because it raises the price of telling the truth and lowers the price of going along.

#### Worked example: the short seller's payoff, and why it was a public service

Why does it matter that short sellers were involved? Because their incentive is the cleanest signal in the market. Walk the math.

Suppose a short seller is convinced Wirecard's shares, trading at €100, are worth close to zero. They borrow 10,000 shares and sell them for €100 each, raising €1,000,000. If the shares fall to €1, they buy 10,000 shares back for €10,000 and return them, keeping roughly €990,000 (before borrowing costs).

But the risk is brutal and asymmetric. If they are wrong and the shares double to €200, they must buy back at €200, paying €2,000,000 to close a position they opened for €1,000,000 — a loss of €1,000,000, *more* than they put up. A short can lose more than 100% of the stake, because a stock can rise without limit.

The one-sentence intuition: **a short seller only takes that asymmetric risk when they are very confident of fraud, which is exactly why their loud warnings deserve investigation, not prosecution.** A regulator that reflexively defends the company against shorts is silencing the one group paid to find the rot.

## BaFin's defense of Wirecard

Germany's financial regulator is called **BaFin** (the Federal Financial Supervisory Authority). Its job is to supervise banks, insurers, and securities markets, and to protect investors and the integrity of the market. In the Wirecard saga it did close to the opposite.

In February 2019, after the *FT*'s whistleblower reporting sent the shares tumbling, BaFin took two extraordinary steps. First, it imposed a **two-month ban on short selling Wirecard shares** — a measure normally reserved for systemic crises, here applied to a single company, framed as protecting the market from manipulation and protecting the wider financial system because Wirecard owned a bank. Second, it filed a **criminal complaint against the *FT* journalists**, alleging market manipulation, on the theory that the reporting was coordinated with short sellers to profit from the share decline.

Sit with that. The regulator, faced with detailed allegations that a DAX company was committing fraud, chose to ban betting against the company and to investigate the reporters. It treated the warning as the crime.

Why? Several reasons, none flattering. There was national pride — Wirecard was Germany's tech champion, and the assumption ran deep that the fraud must be coming from foreign short sellers and an English newspaper, not from a German firm. There was a jurisdictional muddle: BaFin supervised Wirecard *Bank*, the small licensed subsidiary, but the parent holding company was classified as a technology firm, not a financial holding company, so nobody had clear, comprehensive prudential oversight of the whole group. And there was a failure of imagination — the inability to entertain that the respectable, audited, index-listed champion could be a lie.

This is the textbook shape of **regulatory capture** and institutional bias: not bribery, but a regulator that identifies with the entity it is supposed to police, and so defends it. After the collapse, BaFin's president was replaced and the agency was restructured. The lesson the German parliament's inquiry drew was blunt: the regulator had been protecting the company from its critics rather than protecting the public from the company.

The short-selling ban deserves a closer look, because it is the most revealing single act in the whole affair. A short-selling ban is a serious intervention: it tells the market that one side of the trade is forbidden. Regulators reach for it in genuine crises — at the depths of 2008, several countries banned shorting bank shares to stop a self-fulfilling collapse of institutions that were fundamentally solvent. The logic there was that fear, not fraud, was driving the price, and that a temporary ban could break a panic. At Wirecard, the logic was inverted. There was no systemic panic; there was a single company with detailed fraud allegations against it, and the ban had the effect of propping up the share price of a company that was, in fact, a fraud. The regulator used a crisis tool to suppress the market's attempt to price in the truth. Whatever the intention, the effect was to keep the fiction alive — and to let more investors buy in at inflated prices before the reckoning.

The jurisdictional gap is the structural lesson, and it generalizes far beyond Germany. Wirecard the *bank* was supervised by BaFin's banking arm. Wirecard the *group* — the holding company that issued the shares and reported the consolidated €1.9bn — was treated as a technology company, supervised, if at all, only as a securities issuer. No single supervisor had a complete, prudential view of the whole entity, with the powers a bank supervisor has to demand documents, inspect, and confirm. This "is it a tech firm or a financial firm?" gap is precisely the gap that the modern wave of fintechs, payment platforms, and crypto exchanges loves to occupy, because regulation that does not know which box you are in tends to leave you in neither. The Wirecard reforms in Germany tried to close the gap by giving BaFin clearer authority over financial holding companies and stronger powers in special audits. But the gap itself — the ambiguity that "technology" companies which move money are not quite banks — remains the soft spot in supervision worldwide.

## The EY audit failure

If BaFin's failure was the most shocking, EY's was the most consequential, because EY was the one with the simplest job: confirm the cash.

EY audited Wirecard for around a decade and issued clean (unqualified) audit opinions year after year, including on the accounts that reported the €1.9bn of escrow cash. An unqualified opinion is the auditor telling the world: we checked, and these accounts give a true and fair view. Investors, lenders, and index providers all relied on it.

The core failure was specific and, in hindsight, almost unbelievable: for years EY did **not obtain the cash balances directly from the banks** that were supposed to hold the escrow money. Instead, it relied on documents — confirmations and screenshots — provided by the trustee and by Wirecard, the very parties with the motive and the means to fake them. When the truth came out, those documents were alleged to be forgeries. The two Philippine banks named as holding the cash, BDO and BPI, stated they had no such accounts and no relationship with Wirecard, and their executives said the documents bearing their names were fake.

The contrast between what should have happened and what did is the difference between a real bank confirmation and the procedure EY relied on.

![A real bank confirmation compared with the procedure EY relied on at Wirecard](/imgs/blogs/wirecard-2020-the-collapse-of-a-fintech-darling-and-the-missing-billions-8.png)

#### Worked example: the bank confirmation that would have ended it in a day

Walk through the procedure that should have happened, and price out how cheap it was.

A cash confirmation works like this. The auditor takes the bank's name and the account from the client's records. The auditor — not the client — sends a written request to that bank: "Please confirm the balance of account X held by Wirecard on 31 December, and reply directly to us." The bank replies directly to the auditor.

The cost of that procedure for two accounts is a few letters, a few phone calls, and perhaps a day of a junior auditor's time — call it the better part of €1,000 of effort. Set that against what it was meant to verify: **€1.9bn**. The ratio of verification cost to the asset at stake was on the order of:

€1,000 ÷ €1,900,000,000 ≈ **0.00005%**.

The one-sentence intuition: **for roughly five-thousandths of one basis point of the asset's value, EY could have asked the banks directly and learned in a day that the money was not there — the cheapest, most basic check in auditing was the one not done, for the largest, most important asset on the books.** When the new auditor (KPMG, brought in for a special review in 2019-2020) and then EY itself finally insisted on independent confirmation, the answer came back within weeks: there is no money.

This is why the EY failure is studied alongside Arthur Andersen and Enron. It is not a case of a clever fraud defeating sophisticated procedures. It is a case of the most elementary procedure being skipped on the asset that mattered most, year after year, while the firm collected its audit fees and the company's market value soared past €20bn. EY has said it was the victim of an elaborate and collusive fraud — and there was real forgery involved — but the defense runs straight into the question every first-year auditor is taught: if it is cash, you confirm it with the bank. Directly.

There is a deeper structural problem the Wirecard audit exposes, and it applies to the whole audit profession, not just EY. The auditor is **paid by the company it audits**. EY's fees came from Wirecard. The incentive that creates is subtle but corrosive: an auditor who pushes too hard, who treats a long-standing and prestigious client as a suspect, who insists on procedures that imply distrust, risks losing a lucrative engagement to a competitor who will be more accommodating. Nobody has to be corrupt for this to bend behavior. It is enough that the comfortable path — accepting the client's documents, trusting the relationship built over a decade — is also the profitable one. This is the same conflict that sits inside the [credit rating agencies](/blog/trading/finance/credit-rating-agencies-moodys-sp-fitch), who are paid by the issuers whose bonds they rate, and it is why post-Wirecard reform proposals centered on **mandatory auditor rotation** (forcing companies to change auditors periodically so no relationship grows too cozy) and on **separating audit from consulting** (so the audit firm is not also selling the client millions in advisory work it does not want to jeopardize). The check is only as independent as its incentives allow, and an auditor's incentives are not as independent as the word "independent" suggests.

Professional skepticism is the term auditors use for the mindset they are supposed to bring — a questioning attitude, an alertness to evidence that contradicts the client's story. The Wirecard file is, in retrospect, a catalogue of skepticism not applied: red flags from the press dismissed, confirmations accepted from the trustee rather than demanded from the bank, an asset growing implausibly large in implausibly opaque jurisdictions accepted at face value. Skepticism is hardest to apply precisely where it matters most — to a big, profitable, long-standing client that everyone admires — which is exactly the kind of client a large fraud will be hiding inside.

## How the fiction was sustained for years

A fraud this large does not happen in one night. It is a loop that has to be refreshed every reporting period, because each set of accounts has to balance and each year's growth story has to be told. Seeing the loop makes clear why it was both fragile and durable.

![Pipeline showing how the fake cash story was sustained year after year](/imgs/blogs/wirecard-2020-the-collapse-of-a-fintech-darling-and-the-missing-billions-6.png)

The cycle: book fictional revenue from the opaque partners; claim the matching cash sits in trustee escrow; produce documents — letters, statements, screenshots — as the evidence; have the auditor accept those documents and sign the accounts; and let the clean accounts lift the shares, which raises capital and credibility to fund the next round. Then repeat, with bigger numbers.

It was durable because every layer that could have broken it accepted the documents instead of the cash. It was fragile because the entire loop depended on no one ever picking up the phone to the bank. The moment that happened — the moment the KPMG special audit could not verify the partner cash and EY finally demanded direct confirmation — the loop snapped, and there was nothing underneath.

#### Worked example: how a small annual lie compounds into €1.9bn

Frauds rarely start at €1.9bn. They compound. Suppose the scheme adds €300m of fictional escrow cash in its first big year, and grows that addition 25% a year as the company's "growth" accelerates.

- Year 1: €300m
- Year 2: €300m × 1.25 = €375m, cumulative €675m
- Year 3: €375m × 1.25 ≈ €469m, cumulative €1,144m
- Year 4: €469m × 1.25 ≈ €586m, cumulative €1,730m
- Year 5: a further chunk, cumulative ≈ €1.9bn

The one-sentence intuition: **a fraud that has to keep growing to support an ever-richer valuation compounds the hole until it is too big to ever fill with real cash — which guarantees that, once checked, it cannot be explained away.** A €300m hole might conceivably be papered over. A €1.9bn hole, equal to nearly all the equity, cannot. The growth story that made the shares soar was the same force that made the fraud terminal.

## The June 2020 collapse

By the spring of 2020, the pressure had become impossible to contain. After the *FT*'s 2019 reporting, Wirecard's own supervisory board had commissioned KPMG to conduct an independent **special audit** to clear the air. The KPMG report, published in April 2020, was devastating in its restraint: it said it could **not verify** that the third-party-acquiring revenues and the associated escrow cash were real, because it had not been given the access and the direct confirmations needed. Wirecard spun this as inconclusive rather than damning. The market did not fully buy the spin, but it did not panic either — the shares were volatile but the company was still standing.

Then the end came fast. As EY prepared to sign the long-delayed 2019 annual accounts, it insisted — finally — on independent confirmation of the €1.9bn from the banks. On **18 June 2020**, the trustee arrangement unraveled and the banks said the accounts and the cash did not exist. EY refused to issue an audit opinion, stating there were indications of an elaborate and sophisticated fraud and that it could not confirm the existence of €1.9bn of cash. Without an audit opinion, Wirecard was in breach of its loan covenants and could not certify its accounts.

The share price did what fraud-reveal share prices do.

![Wirecard market value rising to DAX entry then collapsing to near zero](/imgs/blogs/wirecard-2020-the-collapse-of-a-fintech-darling-and-the-missing-billions-3.png)

#### Worked example: the market-cap collapse

Let us price the destruction. At the DAX-entry peak in 2018, Wirecard's market capitalization — the number of shares times the price — was about €24bn (about \$27bn). By the time of insolvency the equity was worth essentially nothing; call it under €0.5bn (about \$0.55bn) before trading was effectively meaningless.

The loss of market value was:

€24bn − ~€0.5bn ≈ **€23.5bn**, a decline of roughly **98%**.

To make it personal: an investor who had put €10,000 into Wirecard shares at the peak and held to the end was left with about:

€10,000 × (0.5 ÷ 24) ≈ **€208**.

The one-sentence intuition: **when the asset underpinning a company is revealed to be fiction, the equity does not fall by the size of the hole — it falls to near zero, because there is no reliable number left to value, and uncertainty itself destroys the price.** The €1.9bn hole did not wipe out €1.9bn of value; it wiped out roughly €23.5bn, because it destroyed trust in *every* number Wirecard had ever reported.

The speed of the final unraveling is itself a lesson. For five years the fraud had withstood public allegations, whistleblower documents, and a special audit. Yet once the single fact — does the cash exist? — was finally forced into the open, the company that had taken years to question took about a week to die. On 18 June EY refused the sign-off. Within a day the company admitted the €1.9bn likely did not exist. The shares fell by more than half in a day, then more than half again. Lenders, who had advanced credit against accounts that were now worthless, moved to call their loans. By 25 June there was no path left, and Wirecard AG filed for insolvency.

Within days, CEO Markus Braun resigned and was arrested. COO Jan Marsalek, who had run the Asian operations at the center of the fraud, disappeared and became an international fugitive — later linked, in reporting, to intelligence services, which gave the whole affair a second, espionage-tinged life. On 25 June 2020, Wirecard AG filed for insolvency, the first DAX company ever to do so. The contrast between the years it took to expose the fraud and the days it took to collapse is the signature of a confidence failure: trust erodes slowly and then disappears all at once, and there is no soft landing once the asset everyone relied on is shown to be imaginary.

## The fallout

The wreckage spread outward through every layer that had failed.

**EY** faced lawsuits and regulatory action. Germany's audit oversight body and prosecutors examined the firm's work; investors sued for the losses they attributed to the clean opinions. The episode became a global argument for audit reform — for mandatory rotation of auditors, for separating audit from lucrative consulting, and for tougher rules on confirming assets. EY maintained it had been deceived by a sophisticated, collusive fraud, but the reputational damage was severe and lasting.

#### Worked example: the audit fee versus what it was meant to protect

Put the gatekeeper-incentive problem in numbers. A large public-company audit might generate, say, €5m a year in fees for the audit firm — a meaningful, recurring, prestigious engagement that nobody on the audit team wants to lose. Now weigh that against what the audit was meant to protect: at the peak, about €24bn of investor market value, resting on the assurance that the accounts were true.

The ratio of the auditor's annual reward to the value riding on its opinion was roughly:

€5m ÷ €24,000m ≈ **0.02%**.

The one-sentence intuition: **the auditor captured a tiny fraction of the value its work protected, but stood to lose its entire fee by being the one to call the client a fraud — an asymmetry that quietly rewards going along and punishes the skepticism the whole system depends on.** Fixing gatekeeping is less about smarter auditors than about realigning who pays them and how easily they can be replaced for asking hard questions.

**BaFin** was overhauled. Its president left, the agency's structure and powers were reformed, and the supervision of payment firms and financial holding companies was tightened so that a group like Wirecard could not again slip between the cracks of "is it a tech company or a financial one?" The German finance ministry faced a parliamentary inquiry into why the regulator had defended the company and pursued the journalists.

**The board and management** faced the criminal courts. Markus Braun stood trial in Munich on charges including fraud and market manipulation, maintaining he too was a victim of others' deception. Jan Marsalek remained at large. The trustee and partner figures in Asia largely vanished from accountability.

**The depositors and the bank.** Because Wirecard owned a licensed bank, there was a real-world ripple: customers who held money with Wirecard-issued cards and accounts faced frozen funds and disruption while the bank was ring-fenced and wound down. This is the thread back to ordinary banking — the moment a payments fiction touched real people's real money. Operational and conduct failures of this kind, and the loss events they generate, are exactly the territory covered in the post on [operational risk, fraud, and cyber loss events](/blog/trading/banking/operational-risk-fraud-cyber-and-the-loss-events).

**The investors.** Index funds that had to hold Wirecard because it was in the DAX took losses on behalf of ordinary savers who had never chosen the stock. Pension funds, retail investors chasing the growth story, and lenders who had extended credit against fictional accounts all absorbed the hit. The €1.9bn that never existed translated into very real losses spread across the financial system.

## Common misconceptions

**"Wirecard was a complicated fraud that only experts could have caught."** No — that is the comforting version, and it is wrong. The fraud rested on the most checkable claim a company can make: that it had €1.9bn of cash in two named bank accounts. The procedure to verify it (a direct bank confirmation) is taught in the first weeks of any audit training and costs almost nothing. The fraud survived because that simple check was not done on the key asset, not because it was too clever to detect.

**"The auditors were fooled, so they are not really to blame."** Forgery did occur, and EY has pressed that point. But the standard for auditing cash is not "did the client hand you a convincing-looking document"; it is "did you confirm the balance directly with the bank". An auditor who relies on client-supplied evidence for the largest asset on the balance sheet has not met the standard, regardless of how good the forgery was. Being deceived by a document you should never have relied on is not exculpatory.

**"The regulator just missed it, like everyone else."** BaFin did worse than miss it. It actively defended the company, banned short selling of the stock, and filed a criminal complaint against the journalists who were exposing the fraud. The failure was not passive oversight; it was taking the side of the accused against the accusers.

**"Short sellers and the *FT* got lucky."** They were right for *years* before the collapse, with specific, documented, evidence-based allegations. Luck does not look like a five-year paper trail of correct, detailed warnings. What looked like a market attack was, in fact, accurate reporting that the system refused to believe.

**"Wirecard was a one-off — modern accounting and audit prevent this now."** The mechanism — inventing an asset to balance fictional profit, then never independently verifying it — is the same one behind Enron (2001), Parmalat (2003, where €4bn of supposed cash in a Bank of America account did not exist), and Satyam in India (2009). Wirecard is recent, not unique. The gatekeeper failure is a recurring pattern, not a historical curiosity.

## How it shows up in real banks: gatekeeper failures and fintech hype

Wirecard was not a classic bank failure, but it sits in a long line of failures where the people meant to verify the truth did not. The figure below names the gatekeepers and how each one failed — and the pattern repeats across decades.

![Graph of the gatekeepers that failed at Wirecard auditor regulator board market](/imgs/blogs/wirecard-2020-the-collapse-of-a-fintech-darling-and-the-missing-billions-4.png)

**Enron, 2001.** The energy-trading giant used off-balance-sheet vehicles to hide debt and inflate profits, while its auditor, Arthur Andersen, signed clean opinions and, infamously, shredded documents. Andersen, then one of the Big Five accounting firms, collapsed entirely as a result. The lesson Wirecard repeated: an auditor that is too close to the client, or too dependent on its fees, stops being an independent check.

**Parmalat, 2003.** The Italian dairy company is the closest precedent of all. It claimed to hold roughly €4bn of cash and securities in an account at Bank of America in the Cayman Islands. The account did not exist; the confirmation document was forged. The fraud was uncovered when the bank confirmed it had no such account — the exact step that, had it been taken at Wirecard, would have ended the fraud years earlier. Europe had already lived through a €1.9bn-style "cash that isn't there" fraud, and the lesson did not stick.

**Satyam, 2009.** The Indian IT services firm's chairman confessed to inflating cash and profits by over \$1bn for years. Again: fictional assets, an auditor (PwC's Indian arm) that did not catch it, and a board that did not question the founder. The "Enron of India" showed the pattern was global, not Western.

To put Wirecard's hole in the company of other landmark banking failures and scandals, the chart below sets the €1.9bn beside other case studies covered in this series — recognizing these are not all the same kind of number (some are fines, one is a trading loss, one is missing cash), but measured the same way, in billions, the scale is the point.

![Wirecard missing cash compared with other landmark banking failures and fines](/imgs/blogs/wirecard-2020-the-collapse-of-a-fintech-darling-and-the-missing-billions-9.png)

The comparison is instructive. The €1.9bn that vanished at Wirecard is larger than the trading loss that sank Barings in 1995, and in the same league as the industry-wide fines for LIBOR rigging or a single bank's role in the 1MDB heist. But the comparison also flatters Wirecard's hole in one direction and understates it in another. It overstates it in that the LIBOR and 1MDB figures are penalties spread across institutions that survived, whereas the Wirecard number was the wound that killed the company. It understates it in that €1.9bn of *missing cash* destroyed roughly €23.5bn of *market value* — the hole was a tenth the size of the damage it caused, because confidence, once gone, takes everything with it.

**The fintech-hype dimension.** What made Wirecard distinct, and a warning for the present, is the **"fintech" halo**. Wirecard was sold not as a boring payments processor but as a technology disruptor, and technology stories are granted two dangerous privileges: a much higher valuation multiple, and a much lower demand for the boring controls that govern banks. Investors who would have scrutinized a bank's loan book accepted a fintech's "platform" revenue with far less suspicion. The cross-cutting lesson for the wave of payment companies, neobanks, and crypto firms that followed is uncomfortable: **the more a company is celebrated as innovative, the more, not less, you should insist on confirming that its cash, its customers, and its revenue are real.** The collapse of [FTX in 2022](/blog/trading/crypto/ftx-collapse-sam-bankman-fried) — another celebrated, fast-growing financial-technology firm where customer money turned out not to be where it was supposed to be — is the same story in a different costume.

**The deposit-and-payments connection.** Because Wirecard owned a bank and moved real money for real customers, its collapse was also an operational and settlement event, not just an investor loss. When the parent failed, the licensed bank had to be ring-fenced so that customer funds were not dragged into the parent's insolvency. This is the everyday machinery covered in the post on [how the payments business actually moves money between banks](/blog/trading/banking/the-payments-business-how-money-actually-moves-between-banks) — and the reminder that behind every "platform" there is, eventually, real money that either is or is not in a real account.

## The takeaway: trust is a chain, and it breaks at its weakest link

The deepest lesson of Wirecard is not about payments, or Germany, or even fraud. It is about the architecture of trust that the entire financial system rests on.

We do not, as investors or depositors or citizens, verify companies ourselves. We cannot. So we delegate that verification to a chain of gatekeepers: the board oversees management, the auditor verifies the accounts, the regulator supervises the market, and the press and short sellers probe for what the others miss. Each link is supposed to be independent, so that if one fails, another catches it. The genius of the design is redundancy. The horror of Wirecard is that **every link failed in the same direction at the same time** — and the redundancy that was supposed to protect us protected the fraud instead.

That is the connection back to the spine of this series. A bank — and Wirecard owned one — is a confidence-funded machine. It works only as long as everyone trusts that the assets are real and that someone competent has checked. Wirecard reveals what happens when that trust is misplaced not in one place but everywhere along the chain: the value does not erode, it vanishes, because confidence is binary. You either trust the numbers or you do not, and once a single forged cash confirmation proves you cannot trust them, you cannot trust *any* of them, and a €24bn company becomes worth nothing in a week.

So how do you use this? If you ever read a company's accounts — as an investor, a lender, an analyst, a journalist — Wirecard hands you three durable habits.

First, **follow the cash, literally.** The most flattering number on an income statement is the easiest to fake; the cash in the bank is the hardest to fake *and the easiest to verify*. Ask: who confirmed this cash, independently, and how recently? If the answer is "the company's own documents", you have learned something.

Second, **treat the people warning you as data, not noise.** When credible, specific, evidenced allegations of fraud are met by the company suing the messenger and the regulator banning the bet against it, the defensiveness itself is information. The system's reaction to a warning often tells you more than the warning.

Third, **distrust the halo.** The richer the growth story and the louder the "innovation", the more the boring controls matter, not less. A fintech that cannot survive a direct phone call to its bank was never a fintech. It was a story.

The €1.9bn that never existed is the number to remember. But the lesson is the chain: trust is only as strong as its weakest, sleepiest gatekeeper — and the whole point of having several is that they are not supposed to fall asleep at once.

## Further reading & cross-links

- [Operational risk, fraud, cyber and the loss events](/blog/trading/banking/operational-risk-fraud-cyber-and-the-loss-events) — the risk category Wirecard belongs to, and how banks measure and provision for fraud and conduct losses.
- [The cards business: issuing, acquiring, interchange and the MDR split](/blog/trading/banking/the-cards-business-issuing-acquiring-interchange-and-the-mdr-split) — how real acquiring economics work, the business Wirecard pretended to run at scale in Asia.
- [The payments business: how money actually moves between banks](/blog/trading/banking/the-payments-business-how-money-actually-moves-between-banks) — the settlement plumbing that, in the real world, leaves the verifiable cash trail Wirecard's escrow story never had.
- [The FTX collapse and Sam Bankman-Fried](/blog/trading/crypto/ftx-collapse-sam-bankman-fried) — the same shape of failure in crypto: a celebrated financial-technology firm where customer money was not where it was supposed to be.
- [Credit rating agencies: Moody's, S&P, Fitch](/blog/trading/finance/credit-rating-agencies-moodys-sp-fitch) — another set of gatekeepers paid by the entities they assess, and what conflicts of interest do to the verification chain.
