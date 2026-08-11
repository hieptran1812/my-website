---
title: "Red flags in the audit report and auditor changes: reading the opinion as a forensic document"
date: "2026-08-11"
publishDate: "2026-08-11"
description: "The audit report is the only document in an annual filing written by someone who is paid to be independent of management — and almost nobody reads it. How to mine the opinion paragraph, the critical audit matters, the going-concern language and the internal-control verdict, and why an auditor walking out is the loudest signal a company can send."
tags: ["forensic-accounting", "audit-report", "auditor-changes", "critical-audit-matters", "going-concern", "material-weakness", "restatements", "wirecard", "pcaob", "sox-404", "internal-control", "financial-statement-fraud"]
category: "trading"
subcategory: "Forensic Accounting"
author: "Hiep Tran"
featured: true
readTime: 48
---

> [!important]
> **TL;DR** — A clean audit opinion is a narrow, technical statement that the financial statements are free of *material* misstatement. It is not a fraud guarantee, not a valuation, and not a promise the company survives the year. Read it for what it actually says, and it becomes one of the most information-dense pages in the filing.
>
> - Four opinions exist — unqualified, qualified, adverse, disclaimer — and two questions pick one: is something *wrong*, or could the auditor *not find out*, and how far does the problem spread.
> - Most of the report is boilerplate. Four blocks carry almost everything: the opinion paragraph, any going-concern paragraph, the Critical Audit Matters, and the tenure-and-signature line at the bottom.
> - Critical Audit Matters (US) and Key Audit Matters (international) are the auditor telling you, in writing, which accounts were hardest to audit. That is a free map to where the estimate risk lives.
> - A company can hold a *clean* opinion on its numbers and an *adverse* opinion on the internal controls that produced them, in the same report, in the same year. These are two separate opinions and most readers never notice the second one.
> - The loudest single signal is not in the report at all: it is a Form 8-K Item 4.01 saying the auditor **resigned**, especially when the replacement is much smaller than the firm that left. Ernst & Young signed roughly a decade of clean opinions on Wirecard AG before refusing to sign the 2019 accounts — and the company filed for insolvency days later.

You can read an entire annual report without ever reading the one page in it that was written by somebody who does not work for the company.

Everything else in a 10-K is management's voice. The revenue number is management's estimate. The MD&A is management's narrative. The footnotes are management's disclosure. Even the "risk factors" are drafted by management's lawyers to be simultaneously frightening and legally useless. There is exactly one document bound into the same filing that comes from an outside party with a professional obligation to disagree with management when the numbers are wrong: the independent auditor's report.

Almost nobody reads it. The universal assumption is that it says the same thing every year, in the same words, and that the only two states of the world are "signed" and "not signed". That assumption is roughly half right — most of the report genuinely is boilerplate — and the other half is where the information is.

![What a clean audit opinion actually asserts, and what it does not](/imgs/blogs/red-flags-in-the-audit-report-and-auditor-changes-1.webp)

The diagram above is the mental model for this entire article. The audit opinion is a *bounded* claim. Inside the boundary sits a genuinely useful assertion: that the statements, taken as a whole, are free of misstatement large enough to change a reasonable reader's decision. Outside the boundary sits almost everything a beginner assumes an audit covers — that no fraud occurred, that the business is sound, that the numbers are *right* rather than merely not-materially-wrong. Confuse the inside for the outside and a clean opinion looks like a clean bill of health. Keep them straight, and the report becomes a forensic document: a record of exactly what an independent professional was and was not willing to put their name to.

This post is about mining that document, and about the moment the document changes hands.

## Foundations: the building blocks of an audit report

Before any of the red flags make sense, five things need defining from zero. If you already know them, skim; if you do not, nothing later works without them.

### Who the auditor is, and who pays them

An **external auditor** is an accounting firm hired to examine a company's financial statements and issue a written opinion on whether those statements fairly present the company's financial position. For a listed company this is not optional — it is a condition of being listed.

The structural oddity, and it matters enormously later, is that **the company pays the auditor**. The audit firm's client is the entity it is supposed to scrutinise. In principle this is managed by having the **audit committee** — a subcommittee of the board made up of non-executive directors — hire, pay and fire the auditor rather than management doing it. In practice, management is in the room, negotiates the fee, and controls the far larger consulting budget the same firm might want. Hold that thought; it explains most of section 8.

In the United States, audits of listed companies are governed by standards written by the **PCAOB** (Public Company Accounting Oversight Board), the regulator created by the Sarbanes-Oxley Act of 2002 in the wake of Enron. Internationally, the equivalent standards are the **ISAs** (International Standards on Auditing), written by the IAASB. The two systems have converged a lot but differ in details that turn out to be forensically useful — a company that reports under both gives you two independent descriptions of the same audit.

### What "opinion" means here

The auditor does not certify the numbers. The auditor expresses an **opinion** — a professional judgment, stated in a standard form of words, about whether the statements are fairly presented. The standard form matters: because the wording is templated, *any deviation from the template is deliberate and means something*. Auditors do not improvise prose. When a sentence appears that is not in the template, a partner and a national technical office argued about it.

### Reasonable assurance, not certainty

The auditor provides **reasonable assurance**, defined as a high but not absolute level of assurance. This is not lawyerly hedging; it is a description of the method. Auditors test *samples*, not populations. They confirm a selection of receivable balances, not every invoice. They re-perform a subset of calculations. A well-executed audit of a company with 4 million transactions might directly examine a few thousand of them.

That sampling design is why an audit is structurally better at catching *big* errors than *small* ones, and better at catching *random* errors than *deliberately concealed* ones. A fraud designed by someone who knows how audits are sampled is a fundamentally different adversary from a bookkeeping mistake. We covered the mechanics of that gap in [how an audit works and what it does not catch](/blog/trading/forensic-accounting/how-an-audit-works-and-what-it-does-not-catch); here we only need the consequence.

### Materiality: the number that decides what "clean" means

**Materiality** is the threshold above which a misstatement could reasonably be expected to change the decision of someone relying on the statements. Everything below it is, formally, allowed to be wrong.

No standard prescribes a formula. In practice firms anchor on a benchmark — a percentage of pre-tax income for a profitable operating company, a percentage of revenue or total assets for a company with volatile or negative earnings — and then apply judgment. A commonly used starting point is roughly 5% of pre-tax income, though the actual figure is a matter of professional judgment and is not disclosed to you.

That last clause is the important one. **You never see the materiality number.** It is in the audit file, not the filing. But you can estimate it, and once you can estimate it, "unqualified opinion" stops being a binary and becomes a quantity.

#### Worked example: how much a company can be wrong by and still get a clean opinion

Take an illustrative company — call it Northline Industrial. All figures here are invented for the arithmetic.

Northline reports revenue of \$800 million, pre-tax income of \$40 million, and total assets of \$1.2 billion.

**Step 1 — planning materiality.** The auditor anchors on pre-tax income at 5%:

\$40,000,000 × 0.05 = **\$2,000,000**

**Step 2 — performance materiality.** Auditors then work to a *lower* threshold when designing individual tests, so that several small undetected errors do not add up past the real line. A typical haircut is 50–75%. At 75%:

\$2,000,000 × 0.75 = **\$1,500,000**

**Step 3 — the trivial threshold.** Below some floor, errors are not even accumulated. A common convention is 5% of materiality:

\$2,000,000 × 0.05 = **\$100,000**

Now suppose the audit finds that Northline recognised \$1,400,000 of revenue a quarter too early. Is that a problem?

- As a share of revenue: \$1,400,000 ÷ \$800,000,000 = **0.175%**
- As a share of pre-tax income: \$1,400,000 ÷ \$40,000,000 = **3.5%**
- Against performance materiality of \$1,500,000: **below it**

The error goes onto a schedule of uncorrected misstatements, the auditor asks management to fix it, management declines because it is immaterial, the auditor agrees it is immaterial, and the opinion is **unqualified**. Nothing about this is improper. It is the system working exactly as designed.

But notice what a reader now knows: for this company, "clean opinion" means "no error we found exceeded roughly \$2 million". A \$1.9 million hole is invisible to the opinion. If you are worried about a \$1.9 million related-party payment, the audit report is not going to help you, and it never claimed it would.

> A clean opinion does not mean the numbers are right. It means no error the auditor found was big enough to be worth arguing about.

### The three filings this post lives in

- The **10-K** (US annual report; the 20-F for foreign private issuers, or the statutory annual report elsewhere) contains the audit report itself.
- **Form 8-K** is the US "something happened" filing, due quickly after a triggering event. Two of its numbered items are the subject of this post: **Item 4.01** (change of accountant) and **Item 4.02** (non-reliance on previously issued financial statements).
- The **proxy statement** (DEF 14A) discloses the audit fee and the non-audit fees paid to the same firm, which is how you measure the independence pressure quantitatively.

That is the whole toolkit. Now the report itself.

## 1. The four opinions, and the two questions that pick one

Ask most people how many kinds of audit opinion there are and they will say two: the good one and the bad one. There are four, and the difference between them is not a severity dial. It is a two-by-two.

![The four opinions: nature of the problem by how far it spreads](/imgs/blogs/red-flags-in-the-audit-report-and-auditor-changes-2.webp)

Two independent questions produce the grid.

**Question one: what kind of problem is it?** Either the auditor *found something wrong* — the statements are misstated — or the auditor *could not find out* — they were unable to obtain sufficient appropriate evidence. These are very different. The first says the numbers are bad. The second says the numbers are unknowable, which is often worse, because it usually means records are missing, a subsidiary would not cooperate, or a balance could not be confirmed with the third party who supposedly holds it.

**Question two: how far does it spread?** Is the problem confined to one account — *material but not pervasive* — or does it contaminate the statements as a whole? **Pervasive** is a term of art: broadly, it means the effect is not confined to specific elements, or if it is confined, it represents a substantial proportion of the statements, or (for disclosures) it is fundamental to a reader's understanding.

Cross the two questions and you get the four opinions:

| | Material, not pervasive | Material **and** pervasive |
| --- | --- | --- |
| **Statements are misstated** | **Qualified** — "except for" | **Adverse** — "do not present fairly" |
| **Could not obtain evidence** | **Qualified** — "except for" | **Disclaimer** — "we do not express an opinion" |

And off the grid entirely sits the **unqualified** (US) or **unmodified** (international) opinion: no material misstatement found, and enough evidence obtained to say so. This is what the overwhelming majority of listed-company reports contain.

Learn the four sentence-stems, because they are what you actually search for:

- **Unqualified:** "In our opinion, the financial statements referred to above present fairly, in all material respects…"
- **Qualified:** "In our opinion, **except for** the effects of the matter described in the Basis for Qualified Opinion paragraph, the financial statements present fairly…"
- **Adverse:** "In our opinion, because of the significance of the matter described…, the financial statements **do not** present fairly…"
- **Disclaimer:** "**We do not express an opinion** on the accompanying financial statements. Because of the significance of the matter described…, we have not been able to obtain sufficient appropriate audit evidence…"

Two practical notes for a reader hunting red flags.

First, **the words "except for" are the single highest-value search string in a filing.** They appear in a qualified opinion and essentially nowhere else in the auditor's standard language. If you only ever run one text search across an annual report, run that one.

Second, a **disclaimer of opinion on a listed company is close to a terminal event**, and it is rare, because a company usually cannot maintain its listing while filing statements that no auditor will opine on. When you see one, the interesting question is almost never "what does this mean" — it is "how long has this been coming, and what did last year's report already say".

### When it breaks

The grid is clean; the application is not. The pressure point is the boundary between "qualified" and "adverse", and between "qualified" and "disclaimer" — the pervasiveness judgment. That judgment is made by the same firm the company pays, and it is not reviewable by you. A qualified opinion where an adverse was arguable looks identical, from outside, to a qualified opinion that was obviously right.

## 2. Anatomy of the report: which paragraphs carry information

Open an audit report and you are looking at roughly one to three pages of highly standardised text. Knowing which blocks are template and which are discretionary turns a five-minute read into a thirty-second one.

![Anatomy of the audit report: which paragraph carries a signal](/imgs/blogs/red-flags-in-the-audit-report-and-auditor-changes-3.webp)

Going down the page:

**The title and addressee.** "Report of Independent Registered Public Accounting Firm", addressed to the shareholders and the board. Pure template — but note that the phrase "Registered Public Accounting Firm" means registered *with the PCAOB*, which is checkable and occasionally interesting.

**The opinion paragraph.** The verdict. Read every word, and specifically check *which statements and which years* are covered. An auditor who joined last year opines on the current year; the prior years in the comparative columns were opined on by somebody else, and that predecessor's report may or may not be reproduced. A company that has changed auditors twice in three years can present three years of statements covered by three different opinions, and any one of them can be the weak link.

**The basis for opinion.** Independence, the standards applied, and the assertion that the evidence obtained was sufficient. Template.

**The going-concern paragraph.** This block only exists if there is substantial doubt. Its *presence is the entire signal*; the wording barely matters. Section 4.

**Critical Audit Matters.** The only place in the report where the auditor volunteers what was difficult. Section 3.

**Management's responsibilities** and **the auditor's responsibilities.** Two blocks of pure template, explaining that management prepares the statements and the auditor obtains reasonable assurance through sampling. Skip both, once you have read them once in your life.

**The tenure line.** In US reports, a sentence reading approximately: *"We have served as the Company's auditor since 2011."* This is a gift. It gives you the length of the relationship in one line, and — more usefully — the *reset year* when it changes. A tenure line that reads "since 2024" on a company founded in 1998 is a question you now have to answer, and section 7 tells you where the answer is filed.

**The signature block.** The firm name, the city, and the date. Three separate signals:

- *The firm.* Which one, and is it the same as last year.
- *The city.* The office that signed. A US-listed company whose entire operation is in one country but whose report is signed in a different city can be routine, or can mean the signing office is relying heavily on component auditors it does not control.
- *The date.* The audit report date is the date the auditor obtained sufficient evidence, and it is close to the filing date. A report dated unusually late relative to the fiscal year end — or a filing preceded by a Form 12b-25 saying the annual report will be late — means the audit ran long. Audits run long for a reason.

## 3. Critical Audit Matters and Key Audit Matters: the auditor's own difficulty map

For most of the history of auditing, the report was pass/fail. It told you the auditor's conclusion and nothing at all about how they reached it, what worried them, or where they had to exercise judgment. Investors complained about this for decades. The regulators eventually agreed.

Internationally, the IAASB introduced **Key Audit Matters** (KAM) under **ISA 701**, requiring auditors of listed entities to describe the matters that, in their professional judgment, were of most significance in the audit. It applies to audits of financial statements for periods ending on or after **15 December 2016**.

In the United States, the PCAOB introduced **Critical Audit Matters** (CAM) in **AS 3101**, phased in over eighteen months: audits of fiscal years ending on or after **30 June 2019** for large accelerated filers, and on or after **15 December 2020** for all other filers.

The exemptions are wider than most readers assume. AS 3101 does not require CAM communication for audits of brokers and dealers reporting under Exchange Act Rule 17a-5, investment companies registered under the Investment Company Act **other than business development companies**, employee stock purchase, savings and similar plans, or **emerging growth companies**. Those auditors may include CAMs voluntarily, but they need not. So if you open a report and find no CAM section at all, check the filer type before concluding the auditor found nothing hard — you may simply be reading a report that was never required to tell you.

The two are similar in spirit and differ in definition. A **CAM** starts from a gateway and then has to clear two further tests. The gateway: the matter was **communicated or required to be communicated to the audit committee**. Then, of the matters that pass through that gate, a CAM is one that:

1. **relates to accounts or disclosures that are material** to the financial statements; **and**
2. involved **especially challenging, subjective, or complex auditor judgment**.

Both conditions must hold. A matter can be difficult without being material, or material without being difficult, and neither is a CAM.

A **KAM** is framed more openly — the matters of *most significance* in the audit, selected from those communicated to those charged with governance. The practical difference is that CAM is narrower and anchored to materiality, so a US report typically carries one or two CAMs while an international report may carry more.

### Why this is the most under-read block in the filing

Think about what those tests actually select for. The auditor is telling you, in a document they sign, which accounts were both material *and* required **especially challenging, subjective, or complex judgment**. That is an insider's answer to the question a forensic reader most wants answered: *where in these statements is the number most dependent on an assumption?*

Estimates are where manipulation lives, because an estimate cannot be caught by a bank confirmation. Revenue recognised over time, expected credit losses, goodwill impairment, the fair value of a Level 3 instrument, the carve-out of a business combination's purchase price — every one of these is a number produced by a model whose inputs management chooses. The CAM tells you which of them the auditor found hardest.

![How to mine a Critical Audit Matter](/imgs/blogs/red-flags-in-the-audit-report-and-auditor-changes-4.webp)

The mining procedure is five steps, and it takes about ten minutes per CAM.

1. **Read the CAM** and note the account it names.
2. **Find that account's balance** in the statements. Write down the number.
3. **Identify the assumption** the CAM says the estimate turns on. The CAM text usually names it explicitly — "the estimated cost to complete", "the discount rate and long-term growth rate", "the probability of collection".
4. **Go to the footnote** where that assumption is quantified and check it against the outside world.
5. **Track it year over year.** This is the step almost nobody does, and it is where the signal is.

#### Worked example: turning a revenue CAM into a testable question

Illustrative figures again, for a company we will call Meridian Systems.

Meridian's audit report carries one CAM: *revenue recognised over time on long-term contracts*, described as involving especially challenging auditor judgment because it depends on management's estimate of total costs to complete each contract.

**Step 2 — the balance.** The balance sheet shows **contract assets** — revenue recognised but not yet billed to the customer — of **\$840 million**, up from **\$280 million** two years earlier.

**Step 3 — the assumption.** Percentage-of-completion. Revenue is recognised in proportion to costs incurred against total estimated costs. Understate the estimate of total cost, and the percentage complete rises, and revenue is pulled forward.

**Step 4 — the footnote.** Check whether the cost-to-complete assumptions moved, and whether there were contract loss provisions.

**Step 5 — the trend.** Now do the arithmetic that makes the CAM actionable.

Over the same two years, revenue went from **\$1,000 million** to **\$1,200 million**.

- Contract assets grew: \$840M ÷ \$280M = **3.0×**
- Revenue grew: \$1,200M ÷ \$1,000M = **1.2×**
- Contract assets as a share of revenue: \$280M ÷ \$1,000M = **28.0%**, rising to \$840M ÷ \$1,200M = **70.0%**

Convert to days, which is easier to feel:

- Two years ago: (\$280M ÷ \$1,000M) × 365 = **102 days** of revenue sitting unbilled
- Now: (\$840M ÷ \$1,200M) × 365 = **256 days**

Unbilled revenue has gone from about three and a half months of sales to about eight and a half months. Every one of those extra days is revenue the company has booked, and the customer has not yet been asked to pay for. That is not proof of anything — long-cycle contracts genuinely do this, and a shift in contract mix explains it innocently — but it is now a *specific* question, aimed at a *specific* account, that you would not have known to ask if you had not read one paragraph of the audit report.

**The intuition:** the CAM is the auditor pointing at the account they found hardest to verify; the year-over-year trend in that account is your test of whether it got harder for a reason.

### What this costs, and when it breaks

CAMs have a well-documented weakness: they became boilerplate faster than anyone hoped. Firms converged on standard language, the same three or four CAM topics dominate across whole industries, and the text often describes *procedures performed* rather than *what was uncertain*. Two consequences for a reader:

- **A CAM's disappearance can be more informative than its presence.** If goodwill impairment was a CAM for three years and vanishes in year four with goodwill still on the balance sheet, something changed — the auditor's assessment, the company's model, or the auditor.
- **A brand-new CAM in a mature company is a change in the auditor's risk assessment**, and the audit committee heard about it before you did.

## 4. Going concern: the ladder, and the rung that hides in a footnote

**Going concern** is the assumption that the entity will continue operating for the foreseeable future rather than being liquidated. It is not a footnote detail — it is the foundation the entire balance sheet rests on. A factory is worth its value in use if the company keeps running and its scrap value if it does not.

Because the assumption is load-bearing, both the company and the auditor have to assess it, and the assessment has a specific vocabulary and a specific window.

![The going-concern ladder: five states, one of which reaches the reader](/imgs/blogs/red-flags-in-the-audit-report-and-auditor-changes-5.webp)

Here is the detail that trips up almost everyone, including people who work in the field: **there are two different clocks, and they do not end on the same day.**

**Management's clock.** Under US GAAP, ASC 205-40 requires management to evaluate conditions and events that raise substantial doubt within **one year after the date the financial statements are issued** (or are available to be issued). Note the anchor: *issuance*, not the balance sheet date. Whatever time the company takes to prepare and file quietly extends this window.

**The auditor's clock.** PCAOB AS 2415 asks the auditor to evaluate whether there is substantial doubt for a reasonable period, **"not to exceed one year beyond the date of the financial statements being audited"**. The anchor here is the balance sheet date.

**The international framing.** IAS 1 sets management's assessment period at "at least, but not limited to, twelve months from the end of the reporting period", and ISA 570 (Revised) does not require the auditor to design procedures reaching beyond the period management assessed.

Take a company with a 31 December year end that issues its statements on 28 February. Management's window reaches **28 February of the following year** — about fourteen months past the balance sheet date. The auditor's window reaches **31 December** — twelve months. Two full months of the company's future sit inside management's evaluation and outside the auditor's.

That gap is not academic. A debt maturity in month thirteen falls inside the window management must evaluate and outside the window the auditor is required to reach. Which means the disclosure obligation and the audit obligation can genuinely come apart, and the place the gap surfaces is the footnote rather than the opinion.

The ladder has five rungs, and only the top one reaches the audit report:

1. **No conditions identified.** Nothing appears anywhere.
2. **Conditions and events identified** — recurring losses, negative operating cash flow, working-capital deficiency, a large near-term maturity, loan covenant breaches. Management may discuss these in MD&A. No formal disclosure is triggered yet.
3. **Substantial doubt raised.** Management concludes the conditions, in aggregate, raise substantial doubt about the ability to continue as a going concern.
4. **Substantial doubt alleviated by management's plans.** If management has plans that are *probable of being effectively implemented* and *probable of mitigating* the conditions, the substantial doubt is considered alleviated. There is a footnote disclosure. **There is no going-concern paragraph in the audit report.** This is the rung readers miss.
5. **Substantial doubt not alleviated.** The footnote must say so using the phrase "substantial doubt", and the auditor's report carries an explanatory paragraph.

Rung four is where the forensic value is. A company at rung four is a company whose auditor agreed there was substantial doubt about its survival — and whose audit report is completely clean. Anyone screening on "did the auditor flag going concern" gets a no. Anyone reading the liquidity footnote gets a yes.

And before treating the absence of a going-concern paragraph as reassurance, read what the auditing standard says about itself. AS 2415 contains this sentence:

> The fact that the entity may cease to exist as a going concern subsequent to receiving a report from the auditor that does not refer to substantial doubt, even within one year following the date of the financial statements, does not, in itself, indicate inadequate performance by the auditor.

That is the standard-setter stating, in its own text, that a company can receive a clean report and fail within the year without the audit having been deficient. It is the most honest sentence in the auditing literature, and it tells you exactly how much weight the absence of a going-concern paragraph will bear.

#### Worked example: a company at rung four

Illustrative figures. Harborline Manufacturing, December year-end, statements issued 28 February.

- Cash at 31 December: **\$60 million**
- Operating cash burn: **\$5 million per month**
- Term loan outstanding: **\$120 million**, maturing **31 January** of the following year

**Step 1 — the runway.** \$60,000,000 ÷ \$5,000,000 per month = **12 months**. Cash is exhausted around 31 December of the following year.

**Step 2 — the two windows.** The statements are issued 28 February, so:

- Management's ASC 205-40 window runs to **28 February of the year after that** — fourteen months past the balance sheet date.
- The auditor's AS 2415 window runs to **31 December of the following year** — twelve months past the balance sheet date.

**Step 3 — the debt lands in the gap.** The \$120 million term loan matures **31 January**, which is month thirteen. That is *inside* management's window and *outside* the auditor's. Projected cash at that date:

\$60M − (13 months × \$5M) = **−\$5M**

The company is projected to have run out of cash a month earlier and to be facing a \$120 million repayment it cannot make.

**Step 4 — what each party must do.** Management cannot ignore the maturity: it falls inside the period it is required to evaluate, so it drives management's substantial-doubt conclusion and its disclosure. The auditor's required evaluation period stops at 31 December — and even ignoring the loan entirely, the operating burn alone exhausts cash right at that boundary, so substantial doubt arises on both clocks here. But notice the general shape: **for a company whose problem sits in month thirteen or fourteen, management's disclosure obligation can bite before the auditor's evaluation period reaches it.**

**Step 5 — management's plans.** Management presents a refinancing commitment letter from its bank syndicate and a plan to raise \$90 million in equity. If the auditor concludes these plans are probable of being implemented and probable of mitigating, the substantial doubt is **alleviated**.

Result: a footnote describing the conditions and the plans, and an **unqualified audit report with no going-concern paragraph**.

**The intuition:** the audit report tells you whether substantial doubt survived management's plans, not whether it existed. The existence is disclosed one document deeper.

### The search strings

Because rung four hides, screen on the text rather than on the report structure. Search the full filing — not just the audit report — for:

- "substantial doubt"
- "going concern"
- "ability to continue as a going concern"
- "management's plans"
- "liquidity" within the same footnote as "covenant"

And note the asymmetry that makes this worth doing: a going-concern paragraph is a *lagging* indicator. It appears when the situation is already visible in the cash flow statement, which we walk through in [reading the cash flow statement](/blog/trading/forensic-accounting/reading-the-cash-flow-statement-why-cash-beats-net-income). If you are learning about the liquidity problem from the audit report, you are learning about it late.

## 5. Internal control: material weakness, significant deficiency, and the second opinion

Here is the fact that surprises most readers: for many US listed companies, the audit firm issues **two opinions**, on two different subjects, in the same report. One is on the financial statements. The other is on **internal control over financial reporting** (ICFR) — the system of processes and checks that produces those statements.

They can disagree.

![Material weakness versus significant deficiency, and the two separate opinions](/imgs/blogs/red-flags-in-the-audit-report-and-auditor-changes-6.webp)

### The severity ladder

Control problems come in three graded sizes, and only the largest is disclosed to you:

- A **control deficiency** exists when a control does not allow management or employees to prevent or detect misstatements on a timely basis. Internal matter. You never see it.
- A **significant deficiency** is a deficiency, or combination of deficiencies, less severe than a material weakness but important enough to merit attention by those responsible for oversight. It is **reported to the audit committee. It is not disclosed publicly.** You never see this one either.
- A **material weakness** is a deficiency, or combination of deficiencies, such that there is a **reasonable possibility that a material misstatement of the financial statements will not be prevented or detected on a timely basis.** This one *is* disclosed — in Item 9A of the 10-K, "Controls and Procedures" — and if one exists at year end, management must conclude that ICFR is **not effective**.

Read that material-weakness definition slowly, because the standard-setters chose every word. It is not "a misstatement occurred". It is "there is a reasonable possibility that a material one **would not be caught**". A material weakness is a statement about the *detection capability of the system*, entirely independent of whether anything actually went wrong this year.

### Why the two opinions can disagree

If the control system cannot be relied on to catch a material misstatement, how can the auditor say the statements are free of material misstatement?

Because the auditor can go around the controls. When controls are unreliable, the auditor performs more **substantive testing** — examining the underlying transactions directly rather than testing the process that produced them. Enough direct testing can support a clean opinion on the numbers even when the system that generated them is broken.

This is the single most under-appreciated combination in financial reporting:

> An unqualified opinion on the financial statements together with an adverse opinion on internal control means the numbers were dragged over the line by hand. It is not a clean year. It is an expensive one.

And it carries a forward-looking implication. Substantive testing at that intensity is not sustainable indefinitely, it is expensive, and it depends on the underlying records being complete enough to test. A material weakness in a *revenue* process is materially more alarming than one in, say, income tax provisioning, because revenue is the account fraud most often attacks — see [revenue recognition games](/blog/trading/forensic-accounting/revenue-recognition-games-channel-stuffing-and-bill-and-hold).

#### Worked example: the price of a clean opinion over a broken control

Illustrative. Calder Software, revenue \$1.2 billion, December year-end.

Management identifies a material weakness: the review control over non-standard revenue contracts did not operate effectively — contracts with unusual terms were approved without the technical accounting review the policy requires.

The auditor cannot rely on that control, so the sampling plan changes:

**Normal year (control relied upon):** test the control's design and operation, then substantively test a sample of roughly **40 contracts**.

**Material-weakness year (control not relied upon):** test **100% of contracts above \$500,000**, which turns out to be **310 contracts** covering **82%** of the revenue balance, plus a statistical sample of the remainder.

The consequences, all illustrative:

- Audit fee: **\$3.2 million → \$4.6 million**, an increase of \$1.4 million, or **+44%**
- Filing timing: the 10-K goes out three weeks later than the prior year
- Opinion on the financial statements: **unqualified**
- Opinion on ICFR: **adverse — not effective**

A screener looking only at the opinion sees a clean year. A reader who opens Item 9A sees a company whose revenue-contract review broke, and whose numbers were validated by brute force.

**The intuition:** when the controls fail and the opinion stays clean, ask what the auditor did instead — and whether they can do it again next year.

### Who is exempt, and why that matters

Not every company gets the second opinion. The auditor attestation on ICFR under Section 404(b) of Sarbanes-Oxley applies to accelerated and large accelerated filers; **non-accelerated filers are outside 404(b)** — though management still has to perform and report its own assessment under 404(a).

The SEC widened that exemption in amendments adopted on **12 March 2020** and effective **27 April 2020**. Under them, an issuer with a public float below **\$700 million** *and* annual revenues of **less than \$100 million** in its most recently completed fiscal year falls into the non-accelerated category, and so out of the auditor-attestation requirement. The same amendments added a check box to the Form 10-K cover page stating whether an auditor attestation on ICFR is included — which is a genuinely useful piece of design, because it means you can answer the scope question from page one.

For a forensic reader this is a scope question you must answer before you conclude anything. A small-cap company with no material weakness disclosed might have excellent controls, or might simply never have had an auditor look at them. Check the cover page of the 10-K — filer status and that attestation check box are both there — before you read the absence of a material weakness as good news.

One more piece of the definition worth having. AS 2201 defines a **significant deficiency** as a deficiency, or combination of deficiencies, "less severe than a material weakness, yet important enough to merit attention by those responsible for oversight of the company's financial reporting". And "reasonable possibility", in the material-weakness definition, is met when the likelihood is either *reasonably possible* or *probable* in the accounting-contingencies sense. That is a low bar. A material weakness does not require that anyone thinks a misstatement is likely — only that one slipping through undetected is more than remote.

## 6. Restatements: Big R, little r, and the one you will never hear about

When previously issued financial statements turn out to be wrong, the error takes one of two roads. They lead to wildly different levels of public visibility, and the quiet road is the busier one.

![Big R versus little r: the restatement fork](/imgs/blogs/red-flags-in-the-audit-report-and-auditor-changes-7.webp)

### The fork

The decision hinges on one question: **was the error material to the previously issued financial statements?**

**If yes — a "Big R" restatement.** The company must file a **Form 8-K under Item 4.02**, titled "Non-Reliance on Previously Issued Financial Statements or a Related Audit Report or Completed Interim Review". This filing states that the board or management has concluded the prior statements **should no longer be relied upon**. It is due within **four business days** of that conclusion. The company then amends the affected filings. This is loud: it generates news coverage, it frequently triggers litigation, and it often coincides with the departure of a CFO.

**If no — a "little r" revision.** If the error was *not* material to the prior periods, but correcting the whole thing in the current period *would* be material to the current period, the company revises the prior-period comparatives when they are next presented. There is no Item 4.02 filing. There is no non-reliance statement. There is usually a footnote, sometimes a short one, sometimes titled something as unremarkable as "Revision of Previously Issued Financial Statements".

The materiality assessment behind that fork is not a pure percentage test. SEC staff guidance is long-established that qualitative factors count — whether the error turns a loss into a profit, whether it moves the company across an analyst consensus, whether it affects a segment the market watches, whether it involves concealment. A 1% error that flips the sign of earnings can be material where a 4% error that does not is immaterial. Which means the fork is a judgment call made by the company, reviewed by its auditor, and disclosed to you only in one direction.

Two further mechanics worth knowing. An Item 4.02 filing is triggered either when the registrant concludes that previously issued statements should no longer be relied upon, **or when the auditor notifies the registrant** that they should not be. And the SEC's guidance is explicit that Items 4.01 and 4.02 cannot be folded into a periodic report the way some other 8-K items can — the general accommodation for a triggering event that falls close to a 10-K or 10-Q expressly excludes these two. They get their own filing, always.

### How to find a little r

You find it by comparison, because nothing announces it. The prior-year column in this year's filing is compared against the same year as *originally reported* in last year's filing. If they differ, something was revised.

#### Worked example: a 21% cut to last year's earnings with no announcement

Illustrative. Ferrowood Group.

In the FY1 annual report as originally filed:

- Revenue: **\$961 million**
- Net income: **\$52 million**

In the FY2 annual report, the FY1 comparative column reads:

- Revenue: **\$946 million**
- Net income: **\$41 million**

Do the arithmetic:

- Revenue difference: \$961M − \$946M = **\$15 million**, or \$15M ÷ \$961M = **1.6%** of revenue
- Net income difference: \$52M − \$41M = **\$11 million**, or \$11M ÷ \$52M = **21.2%**

The revenue reversal was high-margin licence revenue, so the \$15 million of sales carried roughly \$14 million of gross profit, and after tax at 21% the bottom-line effect is \$14M × (1 − 0.21) = **\$11.1 million**.

Now consider the two ways of describing the same event. Against revenue, the error is 1.6% — comfortably arguable as immaterial. Against net income, it is 21% — a number no analyst would call immaterial. The company concluded, and the auditor concurred, that it was not material to the previously issued statements. No Item 4.02 was filed. The correction sat in a footnote.

**The intuition:** the materiality of an error depends entirely on which denominator you divide by, and the company picks the denominator first.

### What this means for a reader

Two habits follow.

**Habit one: always compare the comparatives.** When you open a new annual report, pull the prior year's report alongside it and check that the prior-year column matches what was originally reported. It takes two minutes and it catches every little r.

**Habit two: treat an Item 4.02 as a change of state, not an event.** A Big R restatement tells you the prior statements were wrong *and* that the control environment failed to catch it *and* that the auditor signed the wrong numbers. Expect a material weakness to be disclosed in the same period; if one is not, that is its own question. And check the auditor — because a restatement is the most common precursor to the subject of the next section.

### A caution about restatement statistics

Aggregate restatement counts are widely quoted and easy to misread. 2021 is the cautionary example. According to Audit Analytics' review of that year, there were **1,470 restatements in 2021, a 289% increase** and the highest count since 2006, and **62% were reissuances** — Big R restatements — the highest proportion since 2005.

Read bare, those numbers say US financial reporting fell apart in 2021. Read with one more fact, they say something quite different: **excluding SPACs, the 2021 count was down about 10% year over year**, and the reissuance share excluding SPACs was 24%.

What actually happened is that on **12 April 2021** the SEC staff published a statement on accounting for warrants issued by special purpose acquisition companies, taking the view that certain common warrant terms required *liability* rather than equity classification, with fair-value remeasurement running through earnings. Hundreds of SPACs had used the same template and therefore had to make the same correction at the same time. It was one accounting question, answered once, applied across a cohort.

The forensic lesson generalises beyond 2021: **a spike in restatements can be a governance story or an accounting-guidance story, and the two look identical in a count.** Before reading a restatement statistic as a signal about corporate behaviour, ask whether a single standard-setting event created a cohort. And when you see a Big R restatement at a company you follow, check whether every comparable company restated the same line in the same quarter — because if they did, you are looking at a template, not a tell.

## 7. The auditor change: Form 8-K Item 4.01 and the letter the outgoing firm writes

Everything so far has been about reading a document. This section is about reading a *personnel change*, and it is where the strongest single signal in this entire post lives.

Companies change auditors for entirely ordinary reasons: the fee got too high, the firm's industry expertise no longer fits, mandatory rotation rules require it in some jurisdictions, an acquisition consolidated two audit relationships, the audit committee ran a competitive tender. Most auditor changes are boring.

The regulatory design assumes the rest are not. In the US, a change of accountant triggers a **Form 8-K under Item 4.01**, "Changes in Registrant's Certifying Accountant", and the content requirements come from **Item 304 of Regulation S-K**. What Item 304 forces into the open is unusually well designed, because it was written by people who knew exactly how this disclosure would be gamed.

![What Form 8-K Item 4.01 forces into the open](/imgs/blogs/red-flags-in-the-audit-report-and-auditor-changes-8.webp)

### The five things Item 304 makes the company say

**One: which verb.** The filing must state whether the former accountant **resigned**, **declined to stand for re-election**, or **was dismissed**. This distinction is the whole ballgame, and it is why the section exists.

A dismissal is the company firing the auditor. That is normal. Companies tender audits.

A **resignation** is the auditor firing the *client* — walking away from recurring, contracted revenue. Audit firms are businesses; they do not lightly abandon a paying engagement, and resigning creates its own professional and reputational complications. When a firm resigns, the most common explanations are that it no longer believes it can rely on management's representations, that the work required has outrun what the client will pay, or that the firm's risk committee has decided the client is not worth the exposure. None of those is good news.

"Declined to stand for re-election" is a negotiated middle ground and should be read closer to a resignation than a dismissal.

**Two: the two-year opinion history.** Whether the former accountant's reports on the last two years contained an adverse opinion or a disclaimer, or were qualified or modified. This closes the loophole where a company changes auditors and hopes you do not go back and read the old reports.

**Three: disagreements.** Whether there were any **disagreements** with the former accountant, during the two most recent fiscal years and the subsequent interim period, on any matter of **accounting principles or practices, financial statement disclosure, or auditing scope or procedure**, which if not resolved would have caused the former accountant to make reference to the matter in its report.

The overwhelming majority of Item 4.01 filings answer this with a flat no. That is what makes a **yes** so loud. A disclosed disagreement means an accounting dispute was serious enough that the outgoing firm would have qualified its report over it — and the company changed firms.

Note also how broadly the term is defined. The instructions to Item 304 direct that "disagreements" be read to include **any difference of opinion** on those matters — it does not require a formal dispute, a letter, or a threatened resignation. A company answering "none" is therefore making a fairly strong claim.

**Four: reportable events.** A separate category from disagreements, with four enumerated kinds. The former accountant having advised the registrant that:

- **(A)** internal controls necessary to develop reliable financial statements **did not exist**;
- **(B)** it was **unable to rely on management's representations**, or unwilling to be associated with the financial statements;
- **(C)** it needed to **expand significantly the scope** of its audit, or information came to its attention that if further investigated might materially affect the fairness or reliability of the statements, or might cause it to become unwilling to rely on management's representations;
- **(D)** information came to its attention that it concluded **materially impacts the fairness or reliability** of a previously issued audit report or of the financial statements.

Read (C) again, because it is about as explicit a warning as a regulatory form can contain: *information came to the outgoing auditor's attention that it did not get to investigate before leaving.* And read (D) alongside it — that is a former auditor saying a report it already signed can no longer be trusted.

**Five: the exhibit letter.** The company must request a letter from the **former accountant**, *addressed to the Commission*, stating whether it **agrees** with the statements the company made in the filing and, if not, the respects in which it does not agree. That letter is filed as **Exhibit 16**.

The timing rules matter, because they create a gap you can watch. If the letter is not available when the 8-K is filed, the company must file it by amendment **within ten business days** of the original filing — or **within two business days of receiving it**, if it arrives later than that. So a bare Item 4.01 with no Exhibit 16 attached is not necessarily a problem, but it *is* an open question with a deadline on it. Diary the date. If the amendment appears on day ten rather than day one, the letter took nine days to write, and one-sentence letters do not take nine days.

This is the most elegant piece of disclosure design in the whole item. The company writes the narrative; the auditor who just left gets to tell the regulator, in writing and on the public record, whether that narrative is accurate. Three outcomes, in ascending order of interest:

- The letter agrees. Standard, and it is what almost every Exhibit 16 says — usually in a single sentence.
- The letter agrees *only in part*, or declines to comment on portions.
- The letter **disagrees**, or states the former accountant is not in a position to agree.

A two-sentence Exhibit 16 letter is a non-event. A letter that runs to several paragraphs, or that carves out specific statements, is a former auditor choosing their words very carefully, and it deserves to be read as such.

### Reading the successor

Item 4.01 also covers the *new* accountant, including whether the company consulted the new firm on any specific accounting matter before engaging it. That question exists to catch **opinion shopping** — asking prospective firms how they would treat a contested item and hiring whichever gives the answer management wants.

The disclosure requirement makes bare opinion shopping harder to do openly. It does not make it impossible to do quietly, which is what the next section is about.

## 8. Auditor shopping and the small-firm downgrade

Suppose a company's auditor has been pushing back on a revenue policy. Management does not want to change the policy. The company cannot simply ask the auditor to sign; the auditor's own risk process will not allow it. What management can do is change the auditor.

The successor firm has an obvious commercial incentive: it is winning a new client. It also has a professional obligation to communicate with the predecessor before accepting the engagement, and the predecessor is required to respond. That guardrail is real and it stops the crudest version of this.

What it does not stop is a **downgrade in audit capability**. The forensic signal is rarely "the company got a friendlier opinion". It is "the company moved from a firm with the resources and independence to argue, to a firm that has neither".

### The size gradient is measurable, not a prejudice

It would be easy to dismiss "big firm good, small firm bad" as snobbery. It is not, and the reason is that the regulator publishes the measurement.

The PCAOB inspects registered audit firms and reports how often it finds a **Part I.A deficiency** — meaning the inspectors concluded the auditor **did not obtain sufficient appropriate evidence to support its opinion**. Note carefully what that does and does not say: it is a finding about the *audit*, not a finding that the financial statements were wrong. A deficient audit can sit under perfectly accurate accounts. What the rate measures is how often the work failed to support the conclusion.

Here is the gradient from the PCAOB's staff update on its **2024 inspection activities**:

| Firm tier | Share of inspected audits with a Part I.A deficiency (2024) | Prior year (2023) |
| --- | --- | --- |
| Big Four US firms | **20%** | 26% |
| Six other US Global Network Firms | **26%** | 34% |
| Eight annually inspected US non-affiliated firms | **52%** | 53% |
| Triennially inspected non-affiliated firms | **61%** | 67% |
| All inspected firms, aggregate | **39%** | 46% |

Read the top and bottom rows together. At the Big Four, roughly one inspected audit in five was found not to support its own opinion. At the small non-affiliated firms inspected once every three years, it was closer to three in five — **about triple the rate**. Every tier improved year over year, and the gradient survived the improvement.

Two things follow.

First, this is the empirical content of "a switch to a much smaller firm is a red flag". You are not guessing that capability fell; you are moving a company from a population where 20% of audits were found unsupported into one where 61% were.

Second, keep the scale in view. The same PCAOB update reports that the Big Four collectively audit roughly **80% of the market capitalisation of US exchange-listed companies** (as of 31 December 2024). So the high-deficiency tiers are, by market value, a small corner of the market — but it is precisely the corner where a company that wants a quiet audit would go looking.

### The counterintuitive part: it is not smallness, it is the client count

A three-partner firm auditing four local private businesses is unremarkable. A three-partner firm signing opinions for a hundred SEC registrants is arithmetically incapable of doing real work on each — and that arithmetic is checkable, because **PCAOB Form AP** names the firm and the engagement partner for every issuer audit.

The SEC's enforcement record contains the definitive example. On **3 May 2024** the SEC charged **BF Borgers CPA PC** and its owner, **Benjamin F. Borgers**, with what it described as deliberate and systemic failures to comply with PCAOB standards, in work incorporated into **more than 1,500 SEC filings between January 2021 and June 2023**. The specifics are worth stating plainly, because they are worse than "sloppy":

- The firm falsely represented that its work would comply with PCAOB standards.
- It **fabricated audit documentation**.
- It falsely stated, in audit reports included in **more than 500** public-company filings, that the audits complied with PCAOB standards.
- Of **369** clients whose filings in that period incorporated the firm's work, **at least 75%** of those filings incorporated non-compliant audits or reviews.

The settlement — entered without admissions or denials — imposed a **\$12 million** civil penalty on the firm and **\$2 million** on Benjamin Borgers, and **permanently denied both the privilege of appearing or practicing before the Commission as accountants**, effective immediately.

Sit with the 369 figure. That is the number a reader could have counted from public data at any point in the preceding three years. The enforcement action told the market something in May 2024 that the client list had been saying since 2021. The forensic point is not that one firm turned out to be fraudulent — it is that **the input to this judgment was a count, available free, that nobody ran.**

When several hundred issuers lose their auditor on the same day, every one of them files an Item 4.01, and every one faces questions about whether its historical filings can be relied upon. A cheap auditor is not cheap.

### The fee signal

Audit fees are disclosed in the annual proxy statement, and they are one of the cleanest quantitative measures available, because an audit fee is roughly proportional to hours worked. A large fee decline that is not explained by a shrinking business means fewer hours. Fewer hours means less testing.

#### Worked example: measuring audit intensity through a change

Illustrative. Two consecutive proxy statements for the same company.

**Before the change:**
- Revenue: **\$900 million**
- Audit fee: **\$4.1 million**
- Fee per \$1 million of revenue: \$4,100,000 ÷ 900 = **\$4,556**

**After the change (new, smaller firm):**
- Revenue: **\$1,170 million** (up 30%)
- Audit fee: **\$1.3 million**
- Fee per \$1 million of revenue: \$1,300,000 ÷ 1,170 = **\$1,111**

The change in audit intensity:

1 − (\$1,111 ÷ \$4,556) = **75.6%**, call it a **76% reduction**

The business grew 30% and the audit shrank by roughly three-quarters per dollar of revenue. There is no benign story in which a larger, more complex company needs a quarter of the audit effort. Either the previous audit was grossly overpriced, or the current one is not being performed to the same standard.

**The intuition:** the audit fee is the closest thing to a public measurement of how much work the auditor is actually doing.

Two cautions so this is not over-read. First, fees fall legitimately — a first-year 404(b) implementation is expensive and the following year is cheaper; a divestiture removes a component audit; a genuinely competitive tender can produce a real single-digit or low-double-digit percentage saving. Second, compare *fee per unit of revenue*, not the raw fee, and compare against industry peers of similar size, because absolute levels vary enormously by sector. The threshold that matters is not "did the fee fall" but "did it fall by more than the business shrank".

## 9. Scoring an auditor change

Individual signals are weak. Combinations are strong. It helps to have a rough, explicit scoring scheme — not because the weights are empirically derived (they are not; they are judgment, and I am labelling them as such), but because writing them down stops you from rationalising away the fourth flag after you have already talked yourself past the first three.

| Signal | Illustrative weight |
| --- | --- |
| Auditor **resigned** (rather than was dismissed) | +3 |
| Auditor was dismissed | +1 |
| "Declined to stand for re-election" | +2 |
| Successor is materially smaller than predecessor | +3 |
| Successor is a micro firm with a large public-client list | +4 |
| **Disagreement** disclosed under Item 304 | +4 |
| **Reportable event** disclosed | +3 |
| Exhibit 16 letter does not fully agree | +4 |
| Change occurs within 60 days of a filing deadline | +2 |
| Second auditor change within three years | +3 |
| Audit fee per unit of revenue falls more than 40% | +2 |

#### Worked example: scoring two auditor changes

**Company A.** The 8-K states the audit committee **dismissed** its Big Four auditor following a competitive tender and engaged a different Big Four firm. No disagreements, no reportable events. The Exhibit 16 letter is one sentence and agrees. No prior change in the last three years. Audit fee moves from \$5.0 million to \$4.4 million on flat revenue, a 12% decline.

Score: dismissal (+1). Everything else scores zero; the fee decline of 12% is below the 40% threshold.

**Total: 1.** This is a tender. Move on.

**Company B.** The 8-K states the auditor **resigned** (+3). The successor is a firm with nine public clients, replacing a Big Four firm (+3 for the downgrade, +4 for the client-concentration profile). The filing discloses a **reportable event** — the former accountant advised that it had become unwilling to rely on management's representations (+3). The Exhibit 16 letter runs three paragraphs and states the former accountant does not agree with portions of the company's description (+4). This is the second auditor change in three years (+3). Audit fee per unit of revenue falls 61% (+2).

Score: 3 + 3 + 4 + 3 + 4 + 3 + 2 = **22**.

**The intuition:** you are not scoring the auditor change. You are scoring how much the outgoing firm was willing to put in writing on its way out.

## Common misconceptions

**"A clean audit opinion means the numbers are correct."** It means no misstatement the auditor *found* exceeded a materiality threshold you never get to see, based on sampling rather than a complete examination. For a mid-cap company that threshold is routinely in the low millions of dollars. Everything smaller can be wrong and the opinion is still clean.

**"If there were fraud, the auditor would have caught it."** The audit is designed to detect material misstatement, and it is far better at error than at concealment. Collusion, forged third-party confirmations, and management override of controls are the specific scenarios auditing standards themselves acknowledge as hardest to detect. The auditor's own responsibilities paragraph says this, in the boilerplate everyone skips.

**"A material weakness means the financial statements are wrong."** It means there is a reasonable possibility a material misstatement *would not be caught*. It is a statement about the detection system, not about this year's numbers. That is why a material weakness routinely coexists with an unqualified opinion — and why the combination is more interesting than either part alone.

**"No going-concern paragraph means the auditor is comfortable with liquidity."** It can equally mean substantial doubt was raised and then alleviated by management's plans, which produces a footnote and no report paragraph. Screen the text of the filing for "substantial doubt", not the structure of the audit report.

**"A restatement always gets announced."** Only a "Big R" restatement triggers an Item 4.02 non-reliance filing. The "little r" revision path corrects prior-period figures in the next set of comparatives with no 8-K and no announcement. You find those by comparing this year's comparative column against last year's originally reported column.

**"Changing auditors is a red flag."** By itself, usually not — tenders and fee negotiations are ordinary corporate housekeeping. What is a red flag is the *shape* of the change: a resignation rather than a dismissal, a downgrade in firm capability, a disclosed disagreement or reportable event, and an Exhibit 16 letter that does not simply agree.

**"Small audit firms are the problem."** Small firms audit small companies competently all the time. The measurable danger sign is a small firm carrying a public-client list that a firm of its size cannot plausibly service — and that ratio is public.

## How it shows up in real markets

### 1. Wirecard and Ernst & Young: the opinion rail against the signal rail

![Two rails: the opinion rail and the signal rail](/imgs/blogs/red-flags-in-the-audit-report-and-auditor-changes-9.webp)

Wirecard AG was a German payments company and, for a period, a member of the DAX index. Ernst & Young served as its auditor for roughly a decade, issuing **ten consecutive unqualified opinions**, the last of them on the 2018 financial statements.

On **18 June 2020**, EY announced it could not verify roughly **€1.9 billion** of cash supposedly held in trustee accounts, and refused to issue an opinion on the 2019 accounts. Days later, on **25 June 2020**, Wirecard filed for insolvency. Chief executive Markus Braun resigned and was arrested that same week; board member Jan Marsalek was dismissed and disappeared, and has been a fugitive since — reporting has linked him to Russian intelligence, though that has not been adjudicated.

The full anatomy of the fraud is a separate story, told in [Wirecard: the missing €1.9 billion](/blog/trading/forensic-accounting/wirecard-the-missing-1-9-billion-euros). What belongs in *this* post is one narrow observation: the audit opinion changed exactly once, at the end, and everything actionable arrived somewhere else first.

Look at what a reader could see, alongside those clean opinions:

- The *Financial Times* published investigations into Wirecard's accounting from **April 2015** onward, including a **30 January 2019** report of whistleblower allegations at the company's Asia-Pacific operations.
- Germany's regulator BaFin responded to that period by banning net short positions in Wirecard shares from **18 February to 18 April 2019**, and criminal proceedings were opened in connection with the reporting. The enforcement attention went to the short sellers and the journalists.
- The supervisory board commissioned a **KPMG special audit**, published **28 April 2020**, which reported that KPMG **could not verify the majority of Wirecard's third-party-acquirer revenue for 2016 to 2018** and had not received sufficient documentation to address all the allegations.

Sit with that third point. Two months before the collapse, a second accounting firm said in a published report that it could not verify the revenue — while the incumbent auditor's ten clean opinions were still on the shelf, none of them withdrawn. Those are not contradictory statements about the world. They are answers to two different questions, and only one of them was the question that mattered.

Afterwards: Germany's audit oversight body **APAS** sanctioned EY's German firm in **April 2023** with a **€500,000** fine, described as the maximum available under the framework applicable at the time, plus a **two-year ban on accepting new audit mandates from public-interest entities** — new engagements only; existing ones continued. Five former Wirecard auditors were fined individually between **€23,000 and €300,000**, and seven more surrendered their licences during the proceedings. The Bundestag inquiry committee delivered its final report on **22 June 2021**, concluding in substance that EY's repeated clean attestations had themselves created a trust effect that helped dispel the fraud allegations.

Civil liability is not settled. As of the sources I could verify (checked 11 August 2026), claims against EY were pending rather than decided — including roughly 280 suits before the Landgericht Stuttgart seeking about €42 million, and a €700 million claim filed at the Landgericht München in December 2023 on behalf of more than 13,000 investors. Braun's criminal trial opened at the Landgericht München I on **8 December 2022**; as of the most recent reporting I could verify (December 2025) it had run more than three years without a verdict. Nothing here should be read as a finding against any individual.

### 2. Steinhoff: the auditor who pushed

On **5 December 2017**, Steinhoff International — a South African-controlled retail group listed in Frankfurt — announced accounting irregularities and the resignation of chief executive Markus Jooste, and appointed PwC to run an independent forensic investigation. The shares fell **58%** in Frankfurt and **56%** in Johannesburg on the first trading day, and roughly **90%** within two weeks.

The auditor's role is the part relevant here, and it is worth stating precisely rather than dramatically: **Deloitte uncovered the irregularities during the course of its 2017 audit and pushed Steinhoff to investigate further.** The accounts were not signed on schedule. PwC's investigation, published in March 2019, ultimately found fictitious and irregular transactions that inflated group income by more than **€6.5 billion** across 2009 to 2017. Deloitte later agreed to contribute up to **R1.3 billion** toward claimants' losses.

The mechanism is section 7's logic seen from the inside. The signal was not a sentence in an audit report — there was no timely audit report. The signal was an *absence with a deadline attached*: the annual accounts did not appear when they were due. For a reader, "the auditor has not signed and the filing is late" is not missing information. It is the information.

### 3. BF Borgers: the signal was a count

The BF Borgers case is set out in section 8 with its figures. The point to carry into a checklist is what preceded the enforcement action: **369 public clients** at a firm of that size, visible in PCAOB filings, for years before **3 May 2024**.

Form AP makes this checkable for any company you hold, in about two minutes. If a company you own is audited by a firm you have not heard of, do not read the firm's website. Count how many other issuers it signs for, and divide by the number of professionals it says it has.

### 4. Carillion: sanctions arrive years later, and they arrive against the auditor

Carillion, a large UK construction and outsourcing group, entered compulsory liquidation on **15 January 2018**. In the years that followed, the UK's Financial Reporting Council fined KPMG **£14.4 million in 2022** and **£21 million in 2023** in connection with its work relating to Carillion. (The two penalties concerned different matters; I am not attributing a specific misconduct to a specific fine here, because I could not verify which was which to the standard this post requires.)

The lesson is about timing, and it is a discouraging one. Auditor sanctions are a *lagging* indicator by years. The £21 million penalty landed roughly five years after the liquidation. Anyone who was waiting for a regulator to confirm that the audit had been inadequate learned it half a decade after the equity was worthless. Regulatory enforcement is how the profession corrects itself; it is not a tool for a reader deciding what to own this quarter.

### 5. The pattern across all four

Every case above shares a structure worth naming explicitly:

- The audit opinion was **clean, or absent, or late** — never *qualified*. Qualified opinions on outright frauds are rare, because a fraud an auditor can specify well enough to qualify is usually one they can specify well enough to resign over. The opinion has two useful states and a fraud rarely produces the middle one.
- The actionable signal appeared in an **adjacent document**: a second firm's special report, an 8-K, a late-filing notification, an unsigned set of accounts, or the arithmetic of a public-client list.
- The **time between the first visible signal and the collapse ran to months or years**, not days. Wirecard's signal rail was lighting up from 2015. That is a great deal of time to act, but only for a reader watching the adjacent documents rather than waiting for the opinion to change.
- **The regulator's confirmation always arrives last.** APAS sanctioned EY in 2023 for audits of 2016 to 2018. The FRC's second Carillion-related penalty came five years after the liquidation. Enforcement is the epilogue, never the warning.

## The checklist: what to pull, in what order

Fifteen minutes, in this sequence, for any company you are considering.

**Pull these five documents:**

1. The latest annual report (10-K, 20-F, or local equivalent) — for the audit report and Item 9A.
2. The *prior* annual report — for the comparative-column check.
3. The latest proxy statement (DEF 14A) — for the audit fee and non-audit fee split.
4. Every Form 8-K under Item 4.01 and Item 4.02 filed in the last three years.
5. The PCAOB Form AP entry for the company, if US-listed — for the firm and engagement partner.

**Then run these checks, in order, stopping when something fails:**

1. **Search the audit report for "except for", "do not present fairly", and "we do not express an opinion".** Any hit means a modified opinion. Stop and read.
2. **Search the whole filing for "substantial doubt".** Hits outside the audit report mean rung four — doubt raised, then alleviated.
3. **Read the Critical Audit Matters or Key Audit Matters.** Note the accounts named. Compare against last year's CAMs: what appeared, what vanished.
4. **Read the tenure line and the signature block.** Same firm as last year? Same city? How late is the date relative to year end?
5. **Open Item 9A.** Is ICFR effective? If not, what is the material weakness, and does it touch revenue or cash?
6. **Compare this year's prior-period column against last year's originally reported figures.** Any difference is a revision — find the footnote that explains it.
7. **Read every Item 4.01 filing.** Resigned or dismissed? Disagreements? Reportable events? Then open **Exhibit 16** and read what the outgoing auditor actually wrote.
8. **Compute audit fee per unit of revenue** for the last three years and compare against the trend.
9. **Count the successor firm's public clients**, if there was a change to a firm you do not recognise.

![The stop-buying combination matrix](/imgs/blogs/red-flags-in-the-audit-report-and-auditor-changes-10.webp)

**And the combinations that should stop you.** No single signal above is a verdict; companies survive material weaknesses, going-concern footnotes, and auditor changes routinely. These specific *pairs* are different, because each pair contains both a problem and evidence that the mechanism meant to catch it has failed:

- **A resignation plus a restatement within the same twelve months.** The auditor left and the numbers were wrong. Whichever came first, the other one explains it.
- **A downgrade from a large firm to a much smaller one, plus a material weakness touching revenue.** The control that produces the most-attacked account is broken, and the capability to audit around it just fell.
- **Going-concern doubt plus a disclosed disagreement over accounting principles.** The company needs its numbers to look a particular way to survive, and its auditor said in writing that it disagreed about how they should look.
- **A late filing notification plus an auditor change plus a new CFO, in any order, inside two quarters.** Three independent participants in the reporting process changed or ran out of time simultaneously.

## When this matters to you

If you hold individual stocks, this is a fifteen-minute annual check on each position, and it is the highest-yield fifteen minutes in fundamental analysis, because the inputs are free, standardised, and almost universally ignored. You will find nothing in the great majority of companies. That is the point — the check is cheap precisely because it is usually negative.

If you work in or near a company's finance function, the same documents read in reverse tell you how your own employer looks from outside. A material weakness disclosed in Item 9A, a large audit fee increase, and a going-concern footnote are all visible to anyone who cares to look, including your competitors and your counterparties.

And if you are learning forensic accounting more broadly, the audit report is the right place to start, because it is the one document where the writer's incentives are at least partially aligned with yours. Everything else in the filing is advocacy. This page is the closest thing to testimony — bounded, hedged, heavily lawyered testimony, but testimony.

The complementary techniques are the numerical ones: [the Beneish M-Score](/blog/trading/forensic-accounting/the-beneish-m-score-detecting-earnings-manipulation) for detecting earnings manipulation from the statements themselves, [the accruals ratio](/blog/trading/forensic-accounting/the-accruals-ratio-and-the-accruals-anomaly) for measuring the gap between earnings and cash, and [the footnotes and MD&A](/blog/trading/forensic-accounting/the-footnotes-and-mda-where-the-bodies-are-buried) for the disclosure that surrounds the audit report. The audit report tells you where an independent professional found the work hard. The ratios tell you whether the numbers behave. Neither is sufficient; together they are a method.

This is educational material about how to read filings, not investment advice.

## Sources & further reading

**Auditing standards and regulatory text**

- PCAOB **AS 3101**, *The Auditor's Report on an Audit of Financial Statements When the Auditor Expresses an Unqualified Opinion* — the US audit report format and the Critical Audit Matter requirement. [pcaobus.org](https://pcaobus.org/oversight/standards/auditing-standards)
- PCAOB **AS 2201**, *An Audit of Internal Control Over Financial Reporting That Is Integrated with An Audit of Financial Statements* — the definitions of material weakness and significant deficiency.
- PCAOB **AS 2415**, *Consideration of an Entity's Ability to Continue as a Going Concern*.
- **AU-C 705 / ISA 705**, *Modifications to the Opinion in the Independent Auditor's Report* — the four-opinion framework and the pervasiveness test.
- **ISA 701**, *Communicating Key Audit Matters in the Independent Auditor's Report* — the international counterpart to CAMs. [iaasb.org](https://www.iaasb.org/)
- **FASB ASC 205-40**, *Going Concern* — the substantial-doubt evaluation and the one-year-after-issuance window.
- **SEC Regulation S-K, Item 304**, *Changes in and Disagreements with Accountants on Accounting and Financial Disclosure* — the content requirements behind Form 8-K Item 4.01, including disagreements, reportable events, and the Exhibit 16 letter.
- **SEC Form 8-K**, Item 4.01 and Item 4.02 — the change-of-accountant and non-reliance triggers.
- **SEC Staff Accounting Bulletin No. 99** (materiality) and **No. 108** (quantifying misstatements) — the basis for the Big R / little r determination.
- **PCAOB Form AP**, *Auditor Reporting of Certain Audit Participants* — the searchable database of engagement partners and firms for every issuer audit. [pcaobus.org](https://pcaobus.org/resources/auditorsearch)

**Cases**

- SEC press release and administrative proceedings concerning **BF Borgers CPA PC and Benjamin F. Borgers**, May 2024. [sec.gov](https://www.sec.gov/)
- SEC settled charges against **Luckin Coffee Inc.**, December 2020 — the \$180 million penalty. [sec.gov](https://www.sec.gov/)
- **Wirecard AG** — the company's 18 June 2020 announcement regarding the €1.9 billion of trustee-account balances and the 25 June 2020 insolvency filing; the KPMG special audit report of April 2020; APAS proceedings against Ernst & Young GmbH.
- **Steinhoff International Holdings N.V.** — the December 2017 announcement of accounting irregularities and the resignation of the chief executive.

**Sibling posts on this blog**

- [How an audit works — and what it does not catch](/blog/trading/forensic-accounting/how-an-audit-works-and-what-it-does-not-catch)
- [The footnotes and MD&A: where the bodies are buried](/blog/trading/forensic-accounting/the-footnotes-and-mda-where-the-bodies-are-buried)
- [The Beneish M-Score: detecting earnings manipulation](/blog/trading/forensic-accounting/the-beneish-m-score-detecting-earnings-manipulation)
- [Reading the cash flow statement: why cash beats net income](/blog/trading/forensic-accounting/reading-the-cash-flow-statement-why-cash-beats-net-income)
- [Revenue recognition games: channel stuffing and bill-and-hold](/blog/trading/forensic-accounting/revenue-recognition-games-channel-stuffing-and-bill-and-hold)
</content>
