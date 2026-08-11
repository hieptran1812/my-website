---
title: "Wirecard: The Missing EUR 1.9 Billion"
date: "2026-08-11"
publishDate: "2026-08-11"
description: "How a DAX-30 payments company reported a cash pile that did not exist, what its own published accounts showed, and why a bank balance is only as real as the confirmation procedure behind it."
tags: ["forensic-accounting", "wirecard", "financial-statement-fraud", "auditing", "bank-confirmation", "cash-flow", "fraud-detection", "germany", "payments", "short-selling"]
category: "trading"
subcategory: "Finance"
author: "Hiep Tran"
featured: true
readTime: 48
depth: "deep-dive"
---

> [!important]
> **TL;DR** — Wirecard did not fail because of a clever accounting estimate. It failed because roughly EUR 1.9 billion of cash that appeared in its accounts was confirmed to its auditor by documents routed through the company and its partners, rather than by the banks themselves.
>
> - The business model was the cover: Wirecard outsourced much of its payment processing to *third-party acquirers* in Dubai, Manila and Singapore, so both the revenue and the cash sat with someone else, in jurisdictions its auditor could not easily reach.
> - The Financial Times reported that those three partners produced about 50% of 2016 revenue and around 95% of EBITDA. A company's profit engine was outside the company.
> - The published accounts were readable. At 31 December 2018 Wirecard reported EUR 2,719.8 million of cash and cash equivalents — 46% of total assets and EUR 797 million more than its entire equity — while bank borrowings rose EUR 399.7 million to EUR 1,466.1 million and the dividend was EUR 0.20 a share, about 7% of earnings.
> - The standard accrual and cash-conversion screens did **not** flag Wirecard. Adjusted operating cash flow of EUR 500.1 million against EBITDA of EUR 560.5 million is an 89% conversion rate. The test that would have flagged it was a different one: *who is holding this cash, and who told the auditor it is there?*
> - The one number to remember: **EUR 1.9 billion**, the balance Wirecard's management board said in June 2020 could not be located, and which the company then said likely did not exist at all.

On 18 June 2020, a German company worth more than Deutsche Bank two years earlier told the market that its auditor would not sign its accounts, because about EUR 1.9 billion of its cash could not be found. Within a week the shares were worth almost nothing, the chief executive had resigned and been arrested, the chief operating officer had disappeared, and Wirecard AG — a member of the DAX 30, Germany's blue-chip index — had filed for insolvency.

The striking thing is not that the money was gone. It is that, on the evidence that has emerged since, the money was never there. And the reason nobody could tell the difference for years comes down to something almost boring: the difference between a bank telling your auditor how much money you have, and *you* telling your auditor how much money you have, on a piece of paper that looks like it came from a bank.

![Two paths to a confirmed cash balance: the auditor asks the bank directly and the bank replies directly, versus the auditor's request routed through the client and a trustee with the bank never in the loop](/imgs/blogs/wirecard-the-missing-1-9-billion-euros-1.webp)

The diagram above is the mental model for this entire post. Two companies can print the identical line on the identical balance sheet — *cash and cash equivalents: EUR 1,900 million* — and one of them is telling the truth while the other is not, and you cannot tell them apart by reading the number. You can only tell them apart by asking what evidence sits behind the number. Cash is the asset every beginner assumes is the safest thing on a balance sheet, precisely because it is the least judgemental: there is no estimate, no useful life, no discount rate, no revenue recognition policy. It is either in the account or it is not.

That assumption is what Wirecard exploited. This post rebuilds the case the way a forensic reader would: first the business model that made the fraud possible, then what the published accounts actually showed, then the confirmation mechanics that are the real heart of the story, then the five years in which the people who were right were investigated and the company was defended, and finally a checklist you can run on any company that reports a large cash balance.

A note on how this post handles facts. This story involves living people. Some have been convicted, some are still before the courts, and some have never been charged with anything. Every figure, date and allegation below is attributed to a source and dated. Where a matter has not been finally decided by a court, it is described as alleged, charged, or reported — not as fact. Where the mechanics need clean arithmetic, the example is explicitly labelled illustrative.

## First principles: what a payments company does, and what "cash" means on a balance sheet

Before any of the forensics makes sense, you need two foundations: how money actually moves when you tap a card, and what the word "cash" is doing on a set of accounts. Neither requires prior finance knowledge. If you already know this material, skim to the worked example at the end of the section — the rest of the post depends on the vocabulary built here.

### The four parties in a card payment

When you buy something online for EUR 100, four institutions are involved and each takes a slice.

- The **issuer** is the bank that gave you your card. It is lending you the EUR 100 (credit card) or debiting your account (debit card). Wirecard did issuing too, through Wirecard Bank AG and Wirecard Card Solutions Ltd.
- The **card scheme** is Visa or Mastercard — the network that routes the message between the issuer and the merchant's side and sets the rules. Schemes are not banks; they are rulebooks with wires attached.
- The **acquirer** is the merchant's bank-side counterparty. It is the entity that is contractually on the hook to the scheme for that merchant's transactions. It "acquires" the payment on the merchant's behalf, and eventually pays the merchant.
- The **payment service provider** or **gateway** is the technology layer that plugs the merchant's checkout page into all of the above: fraud screening, currency handling, retries, reporting.

Wirecard's two reported segments mapped onto this split. *Payment Processing & Risk Management* was the technology and risk layer, and in 2018 it reported revenues of EUR 1,479.9 million and EBITDA of EUR 481.3 million. *Acquiring & Issuing* was the regulated banking side, reporting revenues of EUR 609.3 million and EBITDA of EUR 79.9 million. A third segment, Call Center & Communication Services, was rounding error at EUR 9.1 million of revenue and EUR −0.5 million of EBITDA. (Source: Wirecard AG, Annual Report 2018, key figures.)

Note the shape of that already. The technology segment carried 73% of revenue and 86% of EBITDA. The regulated bank, the part supervised by a banking regulator with a balance sheet and capital rules, was the smaller and much less profitable half. Keep that asymmetry in mind — the fraud lived on the unregulated side.

Two more terms you need:

- **EBITDA** means *earnings before interest, tax, depreciation and amortisation*. It is a rough proxy for the operating cash a business throws off before financing and accounting charges. Wirecard reported group EBITDA of EUR 560.5 million in 2018, up 36.6% (Annual Report 2018).
- A **basis point** is one hundredth of a percentage point, so 0.01%. Payment economics are quoted in basis points because the margins are thin: an acquirer might keep 20 to 100 basis points of the transaction value.

### Why an acquirer holds a reserve — and how that creates a pot of money

Here is the mechanic that made the whole story possible.

If you buy a flight and the airline collapses before you fly, you charge the payment back. A **chargeback** is the scheme forcing the money back out of the merchant's side and returning it to you. The merchant may be gone. Somebody still has to fund it, and under scheme rules that somebody is the acquirer.

So acquirers protect themselves. They hold back a slice of each merchant's takings for a period — typically some percentage of volume for some number of months — and release it once the chargeback window closes. This is a **rolling reserve**, also called a security reserve. Wirecard's own 2018 annual report describes the mechanic plainly in its risk section: "The reserve held by the acquirer serves" as protection against exactly this exposure.

The rolling reserve is where an enormous amount of money accumulates. Take a portfolio of merchants doing EUR 5 billion of annual volume with a 10% rolling reserve held for six months. At steady state that is roughly EUR 250 million sitting in an account, belonging in some economic sense to the merchants, controlled by the acquirer, and available to nobody in the meantime. Multiply across a large book and you have a pool of cash that is real, is large, and legitimately sits somewhere other than the merchant's bank account.

Now add one more layer. When the acquirer is not the company reporting the accounts — when the acquiring is done by a *partner* — the reserve pool sits with the partner. If the reporting company nonetheless claims economic ownership of that pool, someone has to hold it neutrally. That someone is a **trustee**: an independent party that holds an asset on behalf of others under a legal agreement. The account it holds is an **escrow account** — money parked with a neutral third party, released only when agreed conditions are met.

This is a completely normal commercial structure. Escrow exists because it solves a real problem. It also happens to be the single hardest kind of cash for an auditor to verify, for reasons we will get to.

### What "cash and cash equivalents" actually means

On a balance sheet, **cash and cash equivalents** means money in hand or in bank accounts, plus very short-term, very safe instruments that can be turned into a known amount of money almost immediately — typically anything with a maturity of three months or less from acquisition. The definition is deliberately strict, because this is the line every other analysis leans on.

Three properties are supposed to travel with that line, and a forensic reader should test each one separately:

1. **Existence.** The money is actually in the account.
2. **Ownership.** The money belongs to the reporting company, not to someone else.
3. **Availability.** The company can spend it. Cash that is pledged, blocked, held for customers, or held in escrow is not the same thing as cash you can use, and good disclosure separates it out as *restricted cash*.

A great deal of forensic accounting on cash comes down to noticing that a company has quietly conflated these three. Money that exists but is not yours, or is yours but cannot be moved, is not the same asset as free cash — and it should not be read as if it were. This is the same discipline covered in [reading the balance sheet](/blog/trading/forensic-accounting/reading-the-balance-sheet-what-companies-hide-here) and, from the flow side, in [why cash beats net income](/blog/trading/forensic-accounting/reading-the-cash-flow-statement-why-cash-beats-net-income).

Wirecard's own accounts show how easily property three gets muddied even in ordinary reporting. Its 2018 liquidity discussion notes that Wirecard Bank AG and Wirecard Card Solutions Ltd "hold customer deposits from the banking and card business" — money that is on the balance sheet but is emphatically other people's money. The report goes on to explain that it presents a second, adjusted operating cash flow specifically "in order to eliminate these items," because the raw figure is distorted by the deposit business (Annual Report 2018, liquidity analysis).

That is a legitimate disclosure. Hold on to it, though, because it establishes something important: Wirecard's readers were already trained to accept that its cash line was complicated, and that management would explain which parts were "really" the company's. Complexity is a place fraud can live, and a business whose cash line genuinely requires explanation is a business where one more layer of explanation does not stand out.

#### Worked example: where the money sits in a EUR 100 card payment

Let us walk one payment through, with round numbers, to fix the vocabulary.

You buy something online for **EUR 100** from a merchant.

1. The issuer authorises and eventually settles **EUR 100** into the payment chain.
2. The card scheme takes its fee. Say **EUR 0.15**.
3. Interchange — the fee that flows from the acquiring side back to the issuer — takes, say, **EUR 0.30**.
4. The acquirer's own margin is, say, **EUR 0.55**.
5. That leaves **EUR 99.00** contractually owed to the merchant: 100 − 0.15 − 0.30 − 0.55 = 99.00.
6. But the acquirer holds a **10% rolling reserve for six months**. So it pays the merchant **EUR 89.10** now and holds **EUR 9.90** back.

Where does each piece live at the end of day one?

| Amount | Who has it | What it is on their books |
| --- | --- | --- |
| EUR 89.10 | Merchant | Revenue, settled |
| EUR 9.90 | Acquirer's reserve account | A liability to the merchant, and cash on the acquirer's balance sheet |
| EUR 0.55 | Acquirer | Revenue |
| EUR 0.45 | Scheme and issuer | Their revenue |

The acquirer's balance sheet shows **EUR 10.45** of cash from this one transaction (EUR 9.90 reserve plus EUR 0.55 margin), but only EUR 0.55 of it is genuinely the acquirer's own money. The other EUR 9.90 is money it owes back.

**The intuition:** in payments, the cash line is structurally inflated by money that belongs to other people, and the honest version of the business depends entirely on the disclosure that separates the two.

## The business model as cover: the third-party acquirer

Now we can describe the arrangement at the centre of the case.

![The third-party acquirer structure: offshore merchants routed through partner acquirers in Dubai, Manila and Singapore, with an escrow account holding rolling reserves and Wirecard booking a share of the economics](/imgs/blogs/wirecard-the-missing-1-9-billion-euros-2.webp)

### What a third-party acquirer arrangement is

Some merchants are difficult to bank. Online gambling, adult content, certain crypto services, high-chargeback verticals, and merchants in jurisdictions where a European acquirer has no licence: these businesses pay well precisely because most acquirers will not touch them.

A **third-party acquirer**, or TPA, is a partner firm that holds the local licences and scheme memberships and does the acquiring itself, while the reporting company supplies technology, referrals, risk management, or simply a commercial relationship — and takes a share of the resulting economics. The arrangement is real and legal. It is how many payment companies enter markets where they are not licensed.

Wirecard's TPA partners were identified in the Financial Times' reporting as **Al Alam Solutions** in Dubai, **PayEasy Solutions** in Manila, and the **Senjo Group** in Singapore. The FT reported that these three partners together accounted for roughly **50% of Wirecard's 2016 revenue and around 95% of its EBITDA**, and that operating profit routed through them across 2016 to 2018 totalled about **EUR 985 million**.

![Two 100% bars showing that third-party acquirer partners accounted for about 50% of Wirecard's 2016 revenue and about 95% of its EBITDA](/imgs/blogs/wirecard-the-missing-1-9-billion-euros-3.webp)

Read that concentration figure slowly, because it is the single most diagnostic fact in the case and it was public well before the collapse. A listed company's *profit* was being produced almost entirely by three firms it did not own, did not consolidate, and whose books its shareholders could not see.

### What the arrangement does to the accounts

The TPA structure has three effects on a set of financial statements, and each one degrades the evidence available to an outsider.

**First, it moves the revenue out of reach.** In a normal acquiring business, the auditor can test revenue by tracing transactions through the company's own systems: here is the merchant contract, here is the transaction file from the scheme, here is the settlement, here is the fee. With a TPA, the transactions run on the *partner's* systems. The company's revenue is a contractual share computed from data the partner supplies. The audit evidence for the top line is, at root, a report from a third party.

**Second, it moves the cash out of reach.** The rolling reserves sit with the partner. If the reporting company claims a share of that pool, the money is held by a trustee somewhere on the partner's side of the world. The company's cash is therefore an amount held by a stranger, in a bank the company does not bank with, under an agreement the company signed with the partner.

**Third, it moves everything into jurisdictions where verification is slow, expensive, and easy to stall.** Dubai, Manila and Singapore are not lawless places. But an auditor sitting in Munich cannot walk into a bank in Manila. It must send a request and wait for a reply, and every step of that process depends on cooperation from people the auditor does not control.

Put the three together and you get a company whose largest revenue source, largest profit source, and largest asset are all evidenced by paper that originates outside the company. The related-party dynamics here rhyme closely with those in [related-party transactions and self-dealing](/blog/trading/forensic-accounting/related-party-transactions-and-self-dealing): the danger is not that the counterparty is unusual, it is that the counterparty's independence is an assertion rather than an observation.

#### Worked example: the same EUR 100 payment through a TPA

Take the same EUR 100 transaction and run it through a TPA arrangement, keeping the same economics but changing who holds what. Numbers are illustrative and chosen for clean arithmetic.

The partner acquirer does the acquiring. It keeps the same **EUR 0.55** margin and holds the same **EUR 9.90** rolling reserve. Under the commercial agreement, the reporting company is entitled to, say, **60% of the acquiring margin** and claims economic ownership of **its proportionate share of the reserve pool** held in escrow.

On the reporting company's books:

- Revenue recognised: 0.60 × 0.55 = **EUR 0.33**
- Cash held on its own behalf at its own banks: **EUR 0.00**
- Claim on escrow, reported within cash resources: 0.60 × 9.90 = **EUR 5.94**

Now ask the auditor's question for each line. For the EUR 0.33 of revenue: what document proves it? A settlement report from the partner. Who produced that report? The partner. For the EUR 5.94 of cash: what document proves it? A balance confirmation for the escrow account. Who holds the account? A trustee. Who introduced the auditor to the trustee? The company.

Scale that single transaction up to a book doing tens of billions of euros of volume, and the EUR 5.94 becomes hundreds of millions, then more than a billion, and the evidence chain does not get any stronger as the number grows. It gets weaker, because the amount is now so large that confirming it properly would be an enormous inconvenience to everyone involved.

**The intuition:** the TPA structure does not merely hide the fraud, it relocates every single piece of audit evidence into the hands of people who are not being audited.

### Why escrow balances are unusually hard to confirm

It is worth being precise about why this particular asset is so difficult, because the difficulty is structural, not a matter of anyone being lazy.

A normal corporate bank account has a clean confirmation path. The company banks with, say, Commerzbank. The auditor sends Commerzbank a standard request on the auditor's own letterhead, to an address the auditor looked up, and Commerzbank replies to the auditor directly. The company is not in the chain except to authorise the disclosure.

An escrow account held by a trustee for the benefit of several parties breaks every part of that:

- **The auditor does not know the bank.** The client does not bank there; the trustee does. The auditor learns of the bank's existence from the client.
- **The account is not in the client's name.** So a standard confirmation addressed to "accounts held by Wirecard" may legitimately return nothing, and the client can explain that away.
- **The legal relationship is a document, not a balance.** Even a genuine bank confirmation of a trustee's account balance does not by itself tell you what share of it belongs to the client — that comes from the escrow agreement, which the client supplied.
- **The natural point of contact is the trustee.** It is administratively easy, and superficially reasonable, to route the request through the trustee. That single convenience is where the control breaks.

The general lesson generalises well beyond this case: **the harder an asset is to confirm independently, the more of your scepticism budget it deserves — and the more likely it is that the confirmation you eventually receive travelled through someone with an interest in the answer.**

## What the published accounts actually said

Here is where the post earns its keep. A great deal of writing about Wirecard implies that the fraud was undetectable from the outside. That is not quite right, and the more useful claim is sharper: **the standard quantitative screens did not catch it, and a small number of structural questions would have.** Let us do both, using Wirecard's own audited FY2018 figures.

![Grouped bars of Wirecard's reported revenue, net income and cash 2015 to 2018 with annotations for the 2018 credit facility, the 2019 SoftBank convertible and the 7% dividend payout](/imgs/blogs/wirecard-the-missing-1-9-billion-euros-4.webp)

The four-year arc, as reported: revenue of **EUR 771 million** in 2015, **EUR 1,028 million** in 2016, **EUR 1,488.6 million** in 2017 and **EUR 2,016.2 million** in 2018 — a near-tripling in three years. Net income across the same years was roughly **EUR 143 million**, **EUR 267 million**, **EUR 260 million** and **EUR 347 million**. Note the shape: the revenue line is smooth and the profit line is not. Net income essentially stalled in 2017, dipping slightly against 2016 while revenue grew 45%, then jumped 34% in 2018. A smooth top line above a lumpy bottom line is not itself suspicious, but it is the sort of thing worth asking about, because it means the margin story changed and somebody should be able to say why.

### The headline numbers

From the Annual Report 2018 key figures page (all EUR millions unless stated):

| Metric | 2018 | 2017 | Change |
| --- | --- | --- | --- |
| Revenues | 2,016.2 | 1,488.6 | +35.4% |
| EBITDA | 560.5 | 410.3 | +36.6% |
| EBIT | 438.5 | 311.5 | +40.8% |
| Earnings per share (basic), EUR | 2.81 | 2.07 | +35.7% |
| Equity | 1,922.7 | 1,640.0 | +17.2% |
| Total assets | 5,854.9 | 4,532.8 | +29.2% |
| Cash flow from operating activities (adjusted) | 500.1 | 375.7 | +33.1% |
| Employees (average) | 5,154 | 4,449 | +15.8% |

Source: Wirecard AG, *Annual Report 2018*, key figures. Percentage changes computed from the reported figures.

And from the net-assets table in the same report:

| Balance sheet line, 31 December | 2018 | 2017 | Change |
| --- | --- | --- | --- |
| Goodwill | 705.9 | 675.8 | +4% |
| Customer relationships | 452.1 | 484.9 | −7% |
| Financial and other assets / interest-bearing securities | 413.6 | 310.2 | +33% |
| Receivables of the acquiring business | 684.9 | 442.0 | +55% |
| Trade and other receivables | 357.4 | 274.7 | +30% |
| Interest-bearing securities and fixed-term deposits | 139.6 | 109.1 | +28% |
| **Cash and cash equivalents** | **2,719.8** | **1,901.3** | **+43%** |
| Total assets | 5,854.9 | 4,532.8 | +29% |

Source: Wirecard AG, *Annual Report 2018*, "Changes in net assets", as at 31 December 2018 and 31 December 2017.

Wirecard also reported that Group interest-bearing liabilities to banks rose by EUR 399.7 million to **EUR 1,466.1 million** (31 December 2017: EUR 1,066.4 million), that it had EUR 1,905.6 million of lending commitments, and that additional credit lines from commercial banks of EUR 436.4 million were available on top.

### The screens that passed

Start with the tests a diligent analyst actually runs, because it matters that they came back clean.

The most common quality-of-earnings screen is **cash conversion**: how much of your reported profit shows up as operating cash. Wirecard's own adjusted operating cash flow was EUR 500.1 million against EBITDA of EUR 560.5 million.

$$\text{Cash conversion}_{2018} = \frac{500.1}{560.5} = 89.2\%$$

The prior year: 375.7 / 410.3 = 91.6%. Both are respectable. A software-and-payments business converting roughly nine tenths of EBITDA into operating cash is not a company that screams at you.

The **accruals** screens behave the same way. The cash-flow-based accruals measure asks how much of net income is *not* backed by cash:

$$\text{Accruals} = \frac{\text{Net income} - \text{Operating cash flow}}{\text{Total assets}}$$

With net income of roughly EUR 347 million (basic EPS of EUR 2.81 on 123,565,586 dividend-entitled shares) and adjusted operating cash flow of EUR 500.1 million, the numerator is *negative* — about EUR −153 million, or −2.6% of total assets. Negative accruals are the conservative direction. On the logic set out in [the accruals ratio and the accruals anomaly](/blog/trading/forensic-accounting/the-accruals-ratio-and-the-accruals-anomaly), this reads as high earnings quality.

Receivables growth is a partial exception and deserves credit as a genuine flag. Receivables of the acquiring business grew 55% against revenue growth of 35.4% — a real divergence. Trade and other receivables grew 30%, slightly slower than revenue. So one of the two receivable lines outran the top line and one did not, which is the kind of mixed signal that a screen produces constantly and that a reader dismisses nine times out of ten. The techniques in [forensic ratios: DSO, DIO, DPO and margin anomalies](/blog/trading/forensic-accounting/forensic-ratios-dso-dio-dpo-and-margin-anomalies) would have surfaced it as amber, not red.

This is the honest and uncomfortable finding: **the ratio toolkit that catches WorldCom and catches channel stuffing does not catch a fabricated bank balance.** Fabricated cash does not create an accrual. It creates an asset that looks like the highest-quality asset there is.

#### Worked example: building the cash-conversion picture from the published accounts

Suppose it is spring 2019, the FY2018 report has just landed, and you want to test whether Wirecard's profits are real. Here is the full calculation you can do from the numbers in the tables above, in five steps.

**Step 1 — Convert EBITDA to cash.** 500.1 / 560.5 = **89.2%**. Pass.

**Step 2 — Check whether the cash balance grew by roughly what the business earned.** Cash rose from 1,901.3 to 2,719.8, an increase of **EUR 818.5 million**. Operating cash flow was 500.1. So the cash pile grew by EUR 318.4 million more than operations generated. Where did the difference come from? Not from investing: the report lists cash outflows for strategic transactions and M&A of EUR 42.5 million, medium-term financing agreements of EUR 115.0 million, internally-generated intangibles of EUR 45.1 million, software of EUR 7.6 million and property, plant and equipment of EUR 23.5 million — EUR 233.7 million of outflow in total. So investing consumed cash rather than producing it.

**Step 3 — Find the financing.** Interest-bearing liabilities to banks rose EUR 399.7 million. There it is. The cash balance grew faster than operations because the company borrowed.

**Step 4 — Ask whether that makes sense.** The company is reporting EUR 2,719.8 million of cash. It borrowed an additional EUR 399.7 million during the year. Borrowing costs money; holding cash in 2018 euros earned close to nothing and could earn less than nothing. A company with EUR 2.7 billion of genuinely available cash does not add EUR 400 million of bank debt unless the cash is not available.

**Step 5 — Check what the company returns to shareholders.** The dividend proposed for 2018 was EUR 0.20 per share, distributing **kEUR 24,713** on 123,565,586 dividend-entitled shares (Annual Report 2018, Report of the Supervisory Board). Against group net income of roughly EUR 347 million:

$$\text{Payout ratio} = \frac{24.7}{347.2} = 7.1\%$$

A company sitting on EUR 2.7 billion of cash, generating EUR 500 million of operating cash flow a year, returned about seven cents of every euro of profit to its owners.

**The intuition:** steps 1 and 3 look fine on their own. It is step 4 — putting the cash balance next to the borrowing decision — that produces the question no honest answer fits.

### The tell that actually works: cash versus behaviour

Let us make step 4 the centrepiece, because it is the transferable technique.

There are three numbers in the FY2018 accounts that cannot comfortably coexist:

1. Cash and cash equivalents of **EUR 2,719.8 million** — that is **46.5%** of the EUR 5,854.9 million balance sheet. Nearly half of everything Wirecard owned was cash.
2. Total equity of **EUR 1,922.7 million**. The reported cash pile exceeded the entire book value of the company by **EUR 797.1 million**. In principle, the shareholders' whole stake could have been funded out of the cash line with EUR 797 million left over.
3. Bank borrowings of **EUR 1,466.1 million**, up EUR 399.7 million in the year, on top of a EUR 1,750 million syndicated revolving credit facility signed on 15 June 2018 with a syndicate of banks led by Commerzbank as agent, and followed in April 2019 by an announced EUR 900 million convertible bond issued in connection with a SoftBank partnership.

Any one of these is unremarkable. Together they describe a company that reported holding more cash than its own book value while simultaneously arranging billions of euros of new financing and paying its owners 7% of earnings.

There is always an explanation for this pattern, and the explanations are often true for honest companies: the cash is trapped in subsidiaries in countries with capital controls, it is regulatory capital in a licensed bank, it is customer money, it is needed for working-capital swings around holidays. Wirecard offered versions of several of these, and its business genuinely had reporting-date volatility, which its 2018 report describes at length.

The forensic move is not to reject the explanation. It is to notice that **the explanation, if true, means the cash is not what the balance-sheet line implies it is** — and then to ask the company to quantify exactly how much is unavailable and where it sits. A company that can answer that crisply is probably fine. A company that answers with the shape of the business rather than a number is telling you something.

### The disclosure that was not there

One more test, and it is the cheapest one in this entire post: search the document.

Search the English text of Wirecard's 2018 annual report for the words **"escrow"** and **"trustee"**. Neither appears. Search for "trust" and you get the ordinary sense of the word — the trust of customers, the trust of shareholders — and nothing else.

The largest single asset class on the balance sheet, and the specific arrangement that would later be the epicentre of the collapse, is not named in the report. The liquidity discussion tells you the remaining funds "were held as deposits with the central bank and demand and short-term fixed-term deposits with banks." It does not tell you that a material part of the group's cash resources was held by a third party on behalf of partners in another hemisphere.

This is the sort of thing you find in [the footnotes and MD&A, where the bodies are buried](/blog/trading/forensic-accounting/the-footnotes-and-mda-where-the-bodies-are-buried) — except that here you find it by its absence. A disclosure gap is evidence. When a company's business model obviously requires a structure, and the report never names the structure, that silence is a finding.

### What the accounts did say about the allegations

The FY2018 report is also a remarkable document to read after the fact, because it addresses the accusations directly and dismisses them.

In the CEO letter, Markus Braun wrote: "With regards to the accusations made by a whistleblower at the site in Singapore, I would like to point out that both our internal investigations and also the independent external audit into the incidents commissioned by us did not uncover any indication of so-called 'round-tripping' or corruption within the conducted audit activities. There were also no material impacts on the net assets, financial position and results of operations."

The risk report, section 2.5, "Summary of investigations in Asia," describes the sequence: in spring 2018 compliance was informed of a whistle-blower's account of activities in the Singapore accounting department; the law firm Rajah & Tann was engaged; a preliminary report was delivered on 4 May 2018. The conclusion: "During the course of the investigations conducted, there has been nothing to confirm either the alleged fraudulent round-tripping payments or the allegations of corruption." The report adds, in a sentence that reads differently now: "The authorities in Singapore are currently still looking into specific allegations. It cannot be ruled out that one or other employees may have committed punishable offences."

The Report of the Supervisory Board records that the consolidated financial statements were "issued with an unqualified audit opinion" by Ernst & Young GmbH Wirtschaftsprüfungsgesellschaft, and that the board "in particular concurs with the conclusion of the auditor that — taking into account the corrections made by Wirecard — there are no objections against the accounting treatment of the facts that were the subject of various allegations made by a purported whistle-blower in Singapore." It notes that the auditor's key audit matters "included allegations by a whistle-blower in Singapore," and that the board had considered "the quality of the alleged behaviour and the materiality threshold for the group audit." At the same 24 April 2019 meeting, the mandates of the CEO and of the COO, Jan Marsalek, were each extended by three years.

For a reader learning forensic technique, this is the passage to study. Everything in it is procedurally correct. Allegations arose; an external law firm was engaged; forensic specialists assisted; the auditor treated it as a key audit matter; the supervisory board discussed it and reached a conclusion; the opinion was unqualified. The process ran, and it produced the wrong answer — a theme developed in [how an audit works and what it does not catch](/blog/trading/forensic-accounting/how-an-audit-works-and-what-it-does-not-catch). Round-tripping, the specific allegation being dismissed here, is the subject of [round-tripping and fabricated revenue](/blog/trading/forensic-accounting/round-tripping-and-fabricated-revenue).

## The forensic heart: how cash is supposed to be confirmed

Everything so far is context. This section is the case.

![A five-rung confirmation ladder from management representation at the bottom to auditor-controlled bank access at the top, with the EUR 1.9 billion marked on the third rung](/imgs/blogs/wirecard-the-missing-1-9-billion-euros-5.webp)

### What a bank confirmation is

An **external confirmation** is audit evidence obtained as a direct written response to the auditor from a third party. For cash, the specific procedure is **bank confirmation**, sometimes called **circularisation**: the auditor writes to every bank the client uses and asks it to state, as at the balance sheet date, the balances on all accounts, any loans and overdrafts, any security or charges over assets, and any accounts the auditor did not know about.

The procedure is old and unglamorous and it works, for one reason: **the auditor controls both ends of it.** The auditor decides which banks to write to, sends the request itself, gives the bank the auditor's own return address, and receives the reply directly. The client's only role is to authorise the bank to disclose. If the client touches the request, the reply, or the address, the evidence is degraded — because at that point the client could have manufactured any part of it.

Auditing standards are explicit that control of the process is the point, not the paper. The auditor is required to maintain control over the confirmation requests and responses, and to treat a response that arrives via the client as something that needs to be checked, not something that is finished.

### The ladder of evidence quality

It helps to rank the ways you can come to believe a bank balance exists, from weakest to strongest. The figure above lays this out as a ladder; here it is in words, with what each rung actually proves.

| Rung | Evidence | What it proves | Who could fake it |
| --- | --- | --- | --- |
| 1 | Management says the cash is there | Nothing beyond management's assertion | Management, trivially |
| 2 | A PDF, screenshot or printout supplied by the client | That a document exists | Anyone with a text editor |
| 3 | A balance confirmation routed through a trustee or agent | That a third party is willing to say so, via a channel the client arranged | The client, in cooperation with the third party |
| 4 | A confirmation the bank sends directly to the auditor, on the auditor's own request | That the bank says the balance exists | Only the bank, or someone who can intercept the auditor's mail |
| 5 | The auditor's own read-only access, reconciled to statements, with interest received visible in the P&L | That the balance exists, is the client's, and behaves like money | Effectively nobody |

The gap between rung 3 and rung 4 is not a matter of degree. It is the difference between evidence the audited party can produce and evidence it cannot. Everything at rung 3 and below is, in the end, the client telling you something with extra steps.

On the reporting and official findings that followed the collapse, the EUR 1.9 billion sat on rung 3. The balances were evidenced by confirmations associated with a trustee arrangement rather than obtained by the auditor directly from the banks that were supposed to hold the money. When the question was finally put to those banks directly in June 2020, the answer came back that the accounts and the documents were not genuine.

#### Worked example: two audit files, same balance

Two companies each report **EUR 1,900 million** of cash and cash equivalents. You are given their audit files. Nothing else about the two companies differs.

**Company A's file contains:**
- A list of 11 banks, compiled by the auditor from the client's prior-year confirmations, the general ledger, and the interest expense schedule.
- 11 confirmation requests on the audit firm's letterhead, with the audit firm's address as the reply-to.
- 11 replies received directly, totalling EUR 1,900 million, each naming the account holder and stating whether the balance is restricted.
- A reconciliation of confirmed balances to the general ledger, with three timing differences explained.
- A cross-check of confirmed average balances against interest income in the P&L.

**Company B's file contains:**
- A trustee agreement supplied by the client.
- A statement of the escrow balance, totalling EUR 1,900 million, provided by the trustee.
- Correspondence with the trustee, initiated through a contact introduced by the client.
- Management's representation that the balances are the group's property.

Both files support the same number. Company A's file survives if the client is dishonest; Company B's file does not survive if the client and the trustee are cooperating. The probability the balance is real is not the same in the two cases, and no amount of care applied *within* Company B's approach fixes it — the defect is the design of the procedure, not its execution.

**The intuition:** an audit file is not a collection of documents supporting a number, it is a chain of custody. Ask who controlled each link.

### The cross-check almost nobody runs: does the money behave like money?

There is one more test, and it is beautiful because it needs no cooperation from anyone. **Real money earns interest, and interest lands in the income statement.**

![Illustrative bar chart of the annual interest a EUR 1,900 million balance would earn at 1%, 2% and 3% deposit rates: EUR 19m, EUR 38m and EUR 57m](/imgs/blogs/wirecard-the-missing-1-9-billion-euros-7.webp)

#### Worked example: the interest-income cross-check

This example is **illustrative arithmetic** — the rates are chosen for clean numbers, not taken from any specific deposit contract.

A balance of **EUR 1,900 million** held for a year earns:

- at 1.0%: 1,900 × 0.010 = **EUR 19 million**
- at 2.0%: 1,900 × 0.020 = **EUR 38 million**
- at 3.0%: 1,900 × 0.030 = **EUR 57 million**

Now open the income statement and find the financial result. If a balance of that size produces no visible interest income, exactly one of four things is true:

1. It is genuinely earning nothing — possible in the euro area in the late 2010s, when policy rates were negative, but much harder to believe for balances held in Asian banks in local or US currency, where deposit rates were clearly positive.
2. The interest is being earned by somebody else — which means the money is not yours, whatever the balance sheet says.
3. The interest is there but has been netted into something else — in which case ask to see the gross figure.
4. The money is not there.

The test does not tell you which. It tells you that a question exists, and it costs you two minutes and a calculator. Then run the same test in reverse on the borrowing side: interest expense on EUR 1,466 million of bank debt is a real cost. A company paying interest on debt while earning nothing on a larger cash pile is destroying money every day it continues, and the only rational explanations are that the cash is restricted, is not the company's, or is not there.

**The intuition:** every balance sheet line has a matching income statement consequence. Fabricated assets are usually silent in the P&L, and that silence is audible if you listen for it.

### Why "the auditor should have caught it" is both true and insufficient

It is easy, afterwards, to say the auditor should have written to the banks. It is also correct — and the criticism levelled at EY in the German parliamentary inquiry and in the subsequent professional-oversight action centres precisely on the failure to obtain confirmations directly from the banks over multiple years.

But a beginner should understand *why* the procedure gets skipped, because the reasons are ordinary and recur everywhere:

- **The arrangement is genuinely unusual, so no standard template fits.** There is no pre-printed form for "confirm the balance in a trustee-held escrow account benefiting several partners."
- **The client is helpful.** Someone volunteers to arrange the contact, chase the reply, translate the document. Helpfulness is indistinguishable from control until you look for the difference.
- **The prior year is a precedent.** Once a procedure has been accepted once, doing it differently implies the earlier year was wrong, and nobody wants to imply that.
- **Materiality reasoning cuts the wrong way.** Very large balances are sometimes given *less* scrutiny per euro, because the auditor's attention goes to areas of estimation and judgement. Cash is supposed to be the easy one.
- **The company is a national champion.** Institutional pressure is not a technical factor, but it is a real one.

The transferable rule is: **the more unusual the asset, the more the auditor's own control of the evidence matters — and unusual assets are exactly where standard procedures do not apply.**

## Who got there first, and what happened to them

The most disturbing part of the Wirecard story is not the fraud. It is the five years in which the fraud was being described publicly and accurately while the institutions responsible for market integrity investigated the people describing it.

![Timeline with an upper track of what journalists and short sellers found from 2015 to 2020 and a lower track of what the institutions did in the same period](/imgs/blogs/wirecard-the-missing-1-9-billion-euros-6.webp)

### The reporting

From April 2015, the **Financial Times** published a series of investigations into Wirecard's accounts, the earliest under the FT Alphaville banner — the "House of Wirecard" series, which by the FT's own archive began on **27 April 2015**. The reporting identified a gap between short-term assets and liabilities in the payment business, and returned repeatedly to the opacity of the group's Asian operations. **Dan McCrum** was the lead reporter across the multi-year investigation.

On **30 January 2019** the FT published material relating to Wirecard's Singapore office, reporting suspected falsification of accounts in the Asia-Pacific region and including allegations from within the company's own accounting function. Wirecard's FY2018 annual report, published a few months later, refers to exactly this episode — the "purported whistle-blower in Singapore" — and reports that the internal and external investigations found nothing to confirm round-tripping or corruption.

Later in 2019 the FT reported on the third-party acquirer arrangement, and on the extraordinary concentration of Wirecard's profit in the three partner firms.

Short sellers were making related arguments across the same period. **Zatarra Research** published a report in **February 2016** alleging serious misconduct; Wirecard's shares fell sharply, and German authorities opened market-manipulation investigations into the report's authors. Whatever one thinks of anonymous short-seller research as a genre — and the [short sellers' playbook](/blog/trading/forensic-accounting/the-short-sellers-playbook-how-activists-find-fraud) covers both its value and its abuses — the direction of the claim was correct.

### The institutional response

Set against that, here is what the institutions did in the same years.

In **September 2018** Wirecard was admitted to the **DAX 30**, Germany's blue-chip index, replacing **Commerzbank**. Index inclusion is mechanical rather than an endorsement, but its practical effect is that every fund tracking the index must buy the stock. The company's implied credibility rose because a rulebook said so.

On **18 February 2019**, following the FT's Singapore reporting and the share-price falls around it, **BaFin**, the German financial supervisor, prohibited the establishment and increase of net short positions in Wirecard shares. The prohibition ran for two months, from **18 February to 18 April 2019**. Banning short selling in a single company's stock is an extraordinary step, and BaFin justified it on market-confidence grounds.

BaFin also referred the FT's reporting to prosecutors, who opened an investigation into the journalists in 2019; German authorities separately investigated short sellers on suspicion of market manipulation. The practical effect — a supervisor referring for investigation journalists and traders who were, on the facts that later emerged, describing something real — is the part that drove the subsequent political reckoning.

In **October 2019**, under pressure, Wirecard's supervisory board commissioned **KPMG** to conduct a special audit intended to lay the allegations to rest. The report, published on **28 April 2020**, did not do that. KPMG reported that it had been unable to verify the majority of Wirecard's revenues and profits from the third-party acquiring business over 2016 to 2018, for want of the documentation it needed. Wirecard's shares fell on the publication.

Sit with that date for a moment, because it is the most damning single fact in the chronology. **Fourteen months before the collapse, an accounting firm hired by the company's own supervisory board reported in public that it could not verify where most of the group's profit came from** — and the shares continued to trade, the DAX membership continued, and the FY2018 accounts continued to carry an unqualified opinion. "We could not confirm it" was on the record long before "it does not exist."

Read the asymmetry plainly. On one side: a newspaper, some short sellers, and an accounting firm hired by the company itself, all reporting that they could not stand behind the numbers. On the other: index inclusion, a short-selling ban, and criminal referrals aimed at the accusers. **The market's formal quality signals all pointed the wrong way, and the informal ones all pointed the right way.**

### A word on how to read this

It would be a mistake to draw the lesson "believe short sellers." Short sellers are wrong often, and some of them are dishonest. The correct lesson is narrower and more useful:

**A regulator's action against a company's critics is not evidence about the company.** It is evidence about the regulator's beliefs. Those are two different things, and conflating them is how a supervisory intervention gets read as a clean bill of health. Wirecard's own annual report leaned on exactly this conflation — investigations conducted, allegations not confirmed, opinion unqualified — and it was all technically true.

## June 2020: the week the balance vanished

EY had audited Wirecard for roughly a decade — from the late 2000s through the FY2018 accounts — and issued unqualified opinions throughout. Then it declined to issue one. The collapse that followed was fast. The core sequence, on the public record:

**18 June 2020.** Wirecard announced that its auditor had indicated it would be unable to issue an audit opinion on the FY2019 consolidated financial statements, because sufficient audit evidence could not be obtained for bank balances on trustee accounts of about **EUR 1.9 billion** — roughly a quarter of the consolidated balance sheet total. The publication of the annual report was postponed. The shares fell precipitously.

**19–22 June 2020.** The two banks in the Philippines whose names appeared on the documentation — **BDO Unibank** and the **Bank of the Philippine Islands** — stated that Wirecard was not a client and that the documents purporting to show the balances were not genuine. **Markus Braun**, the chief executive, resigned. Wirecard then announced that there was a prevailing likelihood that the bank trust account balances of EUR 1.9 billion **did not exist**, and withdrew its prior financial results for FY2019 and the preliminary FY2020 figures, warning that previously published annual accounts might be inaccurate.

**22–23 June 2020.** Braun was arrested by Munich prosecutors on 22 June and released the following day on bail reported at EUR 5 million. He was taken back into custody in July 2020. He has denied wrongdoing and has maintained throughout that he was himself deceived by others inside the company.

**25 June 2020.** Wirecard AG filed for insolvency, citing impending insolvency and over-indebtedness. A DAX-listed company had gone from index membership to insolvency in twenty-one months.

**Jan Marsalek**, the chief operating officer, was dismissed and disappeared shortly before the insolvency filing. He has been the subject of an international search since; reporting on his whereabouts has varied and, as a matter of legal status, he remains a wanted man rather than a convicted one.

The criminal proceedings that followed are the reason this section is written carefully. The trial of Markus Braun and two co-defendants opened at the **Landgericht München I** (the Munich Regional Court) in **December 2022**, on charges reported to include gang-organised commercial fraud, breach of trust, accounting falsification and market manipulation. The proceedings were scheduled to run over more than a hundred trial days.

Three things must be said precisely about that.

First, **charges are charges.** An indictment is the prosecution's account of what it intends to prove, not a finding.

Second, the defendants' accounts conflict. Braun's position has been that he was deceived by others within the company. One former executive, **Oliver Bellenhaus**, who ran the Dubai operation, surrendered to German authorities and has cooperated with prosecutors; his account and Braun's are directly opposed. A court deciding between two such accounts is doing exactly the work that a blog post must not pre-empt.

Third, **as at the date of this post — 11 August 2026 — no verdict is reflected in the sources used here.** The most recent dated proceeding these sources reach is from December 2025. This post therefore asserts no outcome, in either direction, for any defendant. **Nothing here should be read as a conclusion about individual criminal responsibility beyond what a court has decided.**

What is not in dispute, because the company itself said it, is the accounting outcome: the balances were withdrawn, the prior results were disavowed, and the company was insolvent.

## The aftermath: what Germany changed

The institutional failure was severe enough to produce structural reform, which is itself useful evidence about what went wrong.

The **Bundestag** established its third committee of inquiry of that parliamentary term on **1 October 2020**, to examine the roles of the government, the financial supervisor and the auditor. It heard 67 witnesses and delivered its final report on **22 June 2021** (Bundestag Drucksache 19/30900). Its work produced sustained criticism of BaFin's supervisory posture — in particular its treatment of the allegations as a market-manipulation problem rather than an accounting problem — and of the audit. BaFin's president at the time subsequently left the organisation, and the supervisor was restructured and given expanded powers. Separately, **ESMA**, the EU securities regulator, published a fast-track peer review of the German supervisory response in **November 2020**, which is the better-sourced external assessment of BaFin's conduct.

Germany also changed the machinery of enforcement. Before Wirecard, German oversight of financial reporting was two-tier: a private body, the *Deutsche Prüfstelle für Rechnungslegung* (the Financial Reporting Enforcement Panel, DPR or FREP), conducted reviews and referred matters to BaFin only when a company would not cooperate. That arrangement ended. **DPR's mandate was terminated with effect from 31 December 2021, and from 1 January 2022 BaFin became the sole authority for enforcing financial reporting**, with powers to open its own investigations, conduct forensic examinations and communicate publicly about them.

The broader **Financial Market Integrity Strengthening Act** (Finanzmarktintegritätsstärkungsgesetz, or FISG) also tightened the audit regime and auditor liability and strengthened requirements for audit committees. The specific parameters of that regime — rotation periods, the boundary between audit and consulting work for the same client, and the statutory liability caps in § 323 HGB — are worth reading in the legislation itself rather than in any summary, including this one, because the numbers differ by category of negligence and were amended in the process.

The German audit oversight body, **APAS** (Abschlussprüferaufsichtsstelle), acted against **EY**'s German firm and against individual auditors. In **April 2023**, on the reporting of that decision, EY Germany was fined **EUR 500,000** — described as the maximum available under the framework applicable at the time — and prohibited for **two years** from accepting new audit mandates for public-interest entities, meaning listed and similarly significant companies. Five former auditors were fined amounts reported in the range of **EUR 23,000 to EUR 300,000** in connection with the 2016 to 2018 audits, and a further seven surrendered their licences while proceedings were under way.

Two things are worth noticing in that penalty. The first is how small EUR 500,000 is against a EUR 1.9 billion hole — a gap that tells you the sanction regime had not been designed for a failure of this size, which is part of why the law was changed. The second is that the two-year prohibition bites in a way the fine does not: for an audit firm, being unable to win new listed clients is a commercial penalty measured in lost fees rather than in the headline number.

EY has said publicly that it was the victim of an elaborate and sophisticated fraud involving multiple parties and forged documentation, and that even a properly planned and executed audit may not detect a collusive fraud of that kind. On the civil side, claims are **pending rather than decided**: reporting describes roughly 280 suits before the Stuttgart regional court seeking on the order of EUR 42 million in total, and a claim of about **EUR 700 million** filed at the Munich regional court in December 2023 by the shareholder association DSW on behalf of more than 13,000 investors. **No final adverse civil judgment against EY over the Wirecard audits is reflected in the sources used here.**

Two observations for a forensic reader. First, **the reform agenda is a confession about the mechanism**: rotating auditors, separating consulting, and strengthening liability are all responses to the problem of an auditor becoming too comfortable with a long-standing client's unusual arrangements. Second, none of these reforms would have been necessary if the failure had been a genuinely undetectable, one-off deception. Regulators do not rebuild a supervisory architecture over bad luck.

## Common misconceptions

**"Cash is the one line you can trust."** Cash is the line where the *concept* is simplest, which is a different thing. The reliability of the number depends entirely on the confirmation procedure behind it, and that procedure is invisible from outside the audit file. What you can observe from outside is whether the company's behaviour is consistent with holding the cash: borrowing, dividends, interest income, and disclosure quality.

**"A clean audit opinion means the accounts are right."** An unqualified opinion means the auditor obtained what it considered sufficient appropriate evidence that the statements are fairly presented in all material respects. It is a statement about a procedure, made by a firm the company pays, subject to materiality thresholds and to the limits of what audit procedures detect. Wirecard received unqualified opinions for years while the arrangement at the centre of its collapse went unnamed in the report.

**"The regulator would have caught it."** Regulators supervise particular things. BaFin supervised Wirecard Bank AG as a bank; the group holding company was not supervised in the same way, and the enforcement of financial reporting ran through a different mechanism. This gap — a payments group whose regulated banking subsidiary was small relative to the unregulated whole — was one of the structural findings of the aftermath. A regulator's silence is not a clearance.

**"The fraud was undetectable from the outside."** The concentration of profit in three partner firms was published. The absence of any disclosure of the escrow structure was checkable with a text search. The pattern of borrowing while claiming a large cash pile was arithmetic from the annual report. None of this proved fraud — a great many honest companies have odd-looking balance sheets — but it was more than enough to justify refusing to own the shares. The correct standard for an investor is not "can I prove fraud" but "can I explain the accounts."

**"Short sellers caused the collapse."** The share price fell on specific disclosures: the January 2019 reporting, the April 2020 KPMG report, and the June 2020 announcement. Short sellers profit when a price falls, and betting against a company is a claim, not evidence. But conflating the messenger with the cause is precisely the error the institutional response made, and it is worth naming as an error rather than a nuance.

**"Escrow and trustee arrangements are a red flag."** They are not. They are a normal, useful legal structure, and most companies that use them are honest. What is a red flag is an escrow arrangement that is *material*, *undisclosed*, and *unconfirmed by the bank directly*. The three conditions together are the signal; any one alone is noise.

## How it shows up in real markets

Fabricated cash is a recurring pattern, not a Wirecard invention. Studying the family resemblance is more useful than studying any single case.

**Parmalat, 2003.** The Italian dairy group reported a very large balance said to be held at a Bank of America account in the Cayman Islands, in the name of a subsidiary. In December 2003 the bank stated that the document purporting to certify the balance was not genuine, and the group collapsed into insolvency shortly afterwards. The mechanism is the Wirecard mechanism almost exactly: a large balance, an offshore account, a document rather than a direct confirmation. Seventeen years apart, the same procedure failed the same way.

**Satyam Computer Services, 2009.** In January 2009 the chairman of the Indian IT services company disclosed in a letter to the board that the balance sheet carried a very large cash and bank balance that did not exist, accumulated over years alongside overstated revenue. He was subsequently convicted. Again: the fabricated asset was cash, because cash is the asset that does not require a plausible operating story to support it.

**Luckin Coffee, 2020.** The Chinese coffee chain's internal investigation confirmed in 2020 that a substantial amount of reported 2019 revenue had been fabricated, following an anonymous short-seller report that used on-the-ground transaction counting. The cash side matters here too: fabricated revenue must land somewhere, and the somewhere is usually a receivable or a cash balance. See [round-tripping and fabricated revenue](/blog/trading/forensic-accounting/round-tripping-and-fabricated-revenue) for the mechanics of how circular flows manufacture both.

**The general shape.** In each case, the fraud needed three things: a business model complicated enough that a strange structure looked normal; a counterparty outside the auditor's easy reach; and an evidence chain that ran through the company. Remove any one and the fraud is much harder. That is why the checklist below is organised around those three, rather than around ratios.

**Where the pattern does not apply.** It is worth being fair about the base rate. Enormous numbers of companies hold cash offshore, use partners in foreign jurisdictions, and disclose imperfectly, and almost all of them are honest. Forensic accounting is a screening discipline: it tells you where to look harder and what to ask, not what is true. Treating every escrow arrangement as fraud would have you avoiding most of the payments industry.

## The checklist: stress-testing a reported cash balance

Here is the practical residue of the whole case.

![A matrix of seven questions to ask about a reported cash balance, with what a clean answer looks like and what Wirecard's answer looked like](/imgs/blogs/wirecard-the-missing-1-9-billion-euros-8.webp)

### Seven questions

1. **Who physically holds the money?** You want named, regulated banks with which the company itself has a direct relationship. If the answer involves a trustee, a partner, an agent, or "our local arrangements," you are on a lower rung of the ladder.
2. **Who confirmed the balance to the auditor?** You usually cannot see the audit file, but you can ask on a results call, and you can read the auditor's report for what it treats as a key audit matter. A company that cannot say "our banks confirm directly to our auditor" has told you something.
3. **Can the company spend it tomorrow?** Ask for the split between free and restricted cash, in euros. Customer money, regulatory capital, pledged balances and escrow are all legitimate — and all mean the headline number overstates what the company controls.
4. **Is the arrangement even named in the report?** Search the document for the structural words the business model implies: escrow, trustee, restricted, pledged, held on behalf of. A material structure that goes unnamed is a disclosure failure regardless of whether anything else is wrong.
5. **Does interest income match the balance?** Multiply the average balance by a plausible deposit rate and compare with the financial result. A large discrepancy is a question, not an answer, but it is a cheap question.
6. **If the cash is real, why the new debt?** Borrowing while holding a large cash pile costs real money every year. There are honest reasons; make the company state which one, in numbers.
7. **Does profit convert to cash the owners can see?** Dividends and buybacks funded from operating cash flow are the strongest available proof that reported profit is spendable. A trivial payout alongside a large reported cash balance is a combination that deserves an explanation.

### Why the layers all failed

![Four concentric layers of defence — management, internal controls, the external auditor and the regulator — all reporting inward, with journalists and short sellers outside the system](/imgs/blogs/wirecard-the-missing-1-9-billion-euros-9.webp)

The final figure makes the structural point that a checklist alone cannot. Every layer that was supposed to catch this reported, directly or indirectly, into the same system:

- **Management and the board** were, on the allegations and the findings that followed, the source of the problem rather than a check on it.
- **Internal controls and internal audit** report to management. When the fraud is at the top, they are instruments rather than obstacles.
- **The external auditor** is engaged and paid by the company, works from documents the company supplies, and applies materiality thresholds the company can reason about.
- **The regulator** supervised part of the group and treated the accusation as the offence.

The only genuinely independent check came from outside the system entirely — journalists with no commercial relationship to the company, and short sellers with a direct financial interest in being right. Both were investigated for it.

That is not an argument that outsiders are always right. It is an argument about **independence as a structural property**. When you assess whether a check is meaningful, do not ask how competent or well-intentioned it is. Ask who pays it, who supplies its information, and what happens to it if it says no. Applied to Wirecard's four layers, that question would have told you in advance that none of them was capable of catching a fraud run from the top.

## When this matters to you

You will probably never audit a payments company. But the reasoning transfers to anything where you are asked to believe that an asset exists on the strength of a document.

If you invest in individual companies, the practical version is short. Open the annual report. Find the cash line and express it as a share of total assets and of equity. Read the liquidity discussion and the cash-related footnote. Search the document for *restricted*, *pledged*, *escrow*, *trustee*, *held on behalf of*. Then compare the cash balance with three behaviours: the borrowing, the dividend, and the interest income. If the cash is large and the behaviour does not match, you have found something worth an hour — not a fraud, but a question that the company should be able to answer in one number and often cannot.

If you work in or near finance, the transferable idea is the confirmation ladder. Whenever someone hands you evidence, ask which rung it came from and who controlled the chain. A great deal of professional judgement collapses into that one question, and it is the question that would have exposed Wirecard years before it collapsed.

And if you take one thing from the case: **a clean audit opinion, a regulator's intervention on the company's behalf, and membership of a blue-chip index are all statements about process. None of them is evidence about the money.** The money is evidenced by the bank, directly, or it is not evidenced at all.

This is educational material about financial-statement analysis, not investment advice.

To go deeper into the individual techniques this case combines, the closest companions in this series are [how an audit works and what it does not catch](/blog/trading/forensic-accounting/how-an-audit-works-and-what-it-does-not-catch) for the audit mechanics, [related-party transactions and self-dealing](/blog/trading/forensic-accounting/related-party-transactions-and-self-dealing) for counterparties that are not what they appear, [reading the cash flow statement: why cash beats net income](/blog/trading/forensic-accounting/reading-the-cash-flow-statement-why-cash-beats-net-income) and [the cash conversion cycle](/blog/trading/forensic-accounting/the-cash-conversion-cycle-and-what-working-capital-reveals) for the flow-side analysis, [the accruals ratio](/blog/trading/forensic-accounting/the-accruals-ratio-and-the-accruals-anomaly) for the screen that did not fire here and usually does, and [transfer pricing and offshore profit shifting](/blog/trading/forensic-accounting/transfer-pricing-and-offshore-profit-shifting) for why jurisdiction is a variable rather than a detail.

## Sources & further reading

**Primary company filings**

- Wirecard AG, *Annual Report 2018*. Source of every FY2018 and FY2017 figure quoted above: key figures (revenues EUR 2,016.2m, EBITDA EUR 560.5m, EBIT EUR 438.5m, basic EPS EUR 2.81, equity EUR 1,922.7m, total assets EUR 5,854.9m, adjusted operating cash flow EUR 500.1m, average employees 5,154); segment revenues and EBITDA; the net-assets table (cash and cash equivalents EUR 2,719.8m, receivables of the acquiring business EUR 684.9m, trade and other receivables EUR 357.4m, goodwill EUR 705.9m); interest-bearing liabilities to banks of EUR 1,466.1m; the liquidity analysis and the explanation of the adjusted cash flow; the CEO letter's statement on the Singapore whistle-blower allegations; risk report section 2.5, "Summary of investigations in Asia," including the 4 May 2018 preliminary report by Rajah & Tann; and the Report of the Supervisory Board (24 April 2019 meeting), recording the unqualified audit opinion from Ernst & Young GmbH Wirtschaftsprüfungsgesellschaft, the treatment of the whistle-blower allegations as a key audit matter, the dividend of EUR 0.20 per share distributing kEUR 24,713 on 123,565,586 dividend-entitled shares, and the extension of the CEO's and COO's mandates. Available at wirecard.com.
- Wirecard AG, ad-hoc and press releases, June 2020, on the postponement of the FY2019 annual report, the EUR 1.9 billion of trustee-account balances for which sufficient audit evidence could not be obtained, the subsequent statement on the prevailing likelihood that the balances did not exist, the withdrawal of prior results, and the insolvency application of 25 June 2020.
- Wirecard AG, announcement of the SoftBank partnership and the EUR 900 million convertible bond, 24 April 2019; and the EUR 1,750 million syndicated revolving credit facility dated 15 June 2018.

**Journalism and investigative reporting**

- *Financial Times* — the multi-year investigation into Wirecard's accounts led by Dan McCrum, beginning with the FT Alphaville "House of Wirecard" material in 2015, the January 2019 reporting on the Singapore office, and the 2019 reporting on the third-party acquirer arrangement and its contribution to group revenue and EBITDA. The 50% of revenue / 95% of EBITDA concentration figures and the EUR 985 million of operating profit routed through the three partners across 2016–2018 come from this reporting.
- Contemporaneous wire coverage (Reuters, Bloomberg, Associated Press, Deutsche Welle) of the June 2020 sequence: the auditor's refusal, the responses of BDO Unibank and the Bank of the Philippine Islands, Markus Braun's resignation and arrest, and the insolvency filing.

**Regulators, oversight bodies and the legislature**

- BaFin, general administrative act prohibiting the establishment and increase of net short positions in Wirecard AG shares, in force **18 February to 18 April 2019**, and BaFin's contemporaneous statements on the market-manipulation investigations.
- KPMG, special audit report on Wirecard AG, commissioned by the supervisory board in **October 2019** and published on **28 April 2020**, reporting that the majority of third-party acquiring revenues and profits for 2016–2018 could not be verified for want of documentation.
- Deutscher Bundestag, final report of the third committee of inquiry of the 19th Bundestag into the Wirecard case: committee established **1 October 2020**, 67 witnesses heard, report of **22 June 2021**, **Drucksache 19/30900**. This is the citable primary document for the parliamentary findings.
- ESMA, fast-track peer review of the supervisory response to the Wirecard case, **November 2020**.
- Gesetz zur Stärkung der Finanzmarktintegrität (Financial Market Integrity Strengthening Act, FISG), and the transfer of financial-reporting enforcement to BaFin as sole authority from **1 January 2022**, following termination of the Deutsche Prüfstelle für Rechnungslegung's mandate with effect from **31 December 2021**. Read the statute and § 323 HGB directly for the liability caps and rotation parameters.
- Abschlussprüferaufsichtsstelle (APAS), professional-oversight decision concerning the Wirecard audits, reported in **April 2023**: a **EUR 500,000** fine on EY's German firm, described as the maximum available under the applicable framework, and a two-year prohibition on accepting new public-interest-entity audit mandates, with fines reported at EUR 23,000–300,000 on five former auditors and seven licence surrenders during proceedings. EY's public statements in response. APAS is reported to have filed criminal charges in late September 2020 concerning the FY2015–2017 audits.
- Pending civil claims against EY, undecided as at the date of this post: approximately 280 suits before the Landgericht Stuttgart totalling on the order of EUR 42 million, and a claim of about EUR 700 million filed at the Landgericht München in December 2023 by the DSW shareholder association for more than 13,000 investors.
- Criminal proceedings: the trial of Markus Braun and two co-defendants at the **Landgericht München I**, opened **December 2022**, scheduled over more than 100 trial days, on charges reported to include gang-organised commercial fraud, breach of trust, accounting falsification and market manipulation. **No verdict is reflected in the sources consulted for this post, the most recent of which dates from December 2025.**

**For comparison cases**

- Contemporaneous reporting and subsequent court records on Parmalat (December 2003, the disputed Bank of America confirmation), Satyam Computer Services (January 2009, the chairman's disclosure and subsequent conviction), and Luckin Coffee (2020, the internal investigation into fabricated revenue).

A note on the tiering of these sources. Every FY2018 and FY2017 financial figure in this post is taken directly from Wirecard's own published annual report and is quoted as printed — that is primary evidence, and a reader can check it line by line. The legal, regulatory and chronological material is a tier below: it comes from press reporting and from official summaries rather than from the underlying court and regulatory files, and it is dated and attributed on that basis. Where a figure or an outcome matters, go to the primary document named above rather than relying on this account of it. Where this post describes matters that were still before the courts as at the date of the sources cited, it says so. Allegations are described as allegations, and nothing here should be read as a finding of individual criminal responsibility beyond what a court has decided.
