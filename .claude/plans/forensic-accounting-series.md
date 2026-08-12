# Forensic Accounting — "Cooking the Books" Blog Series Plan

**Subcategory:** `trading/forensic-accounting/`
**Display label:** "Forensic Accounting"
**Size:** 40 posts, 8 waves (5 posts/wave)
**Angle:** A field guide to **how companies manipulate their financial statements — and how to catch them.** Teaches a curious beginner to (1) actually *read* the three statements and the cash flow, (2) recognize every major trick in the manipulator's toolkit and the *purpose* of each move, (3) apply a concrete detection toolkit (M-Score, Z-Score, accruals, forensic ratios, Benford, audit red flags, the short-seller playbook), and (4) understand it all through clear, sourced case studies (Enron, WorldCom, Wirecard, Luckin, Satyam, Toshiba, Lehman, Steinhoff, FLC). Every post moves **intuition → the statement mechanics → the trick shown on real statement lines with dollar figures → why they did it → how to detect it.**
**Depth:** all `deep-dive` (`featured: true`).
**Voice:** accessible forensic-accountant / investigator — precise, plain-English, no gatekeeping, no fabrication. Sibling of Analyst's Edge, Risk Management, Trading Psychology.
**Language:** English only (verify gate enforces). User briefed in Vietnamese; posts ship in English, Vietnam cases covered in English.

## Conventions (mirror Trading-Psychology / Analyst's Edge)
- Tool: **finance-writer** skill. TL;DR `> [!important]` callout at top; a Foundations/mechanism H2 (must contain a gate trigger word: "Foundations"/"First principles"/"how it works"/"the building blocks"/"fundamentals"); ≥4 `#### Worked example:` walkthroughs with real numbers (show the actual statement lines / journal entries being manipulated); ≥1 named real-company case; **fact-check Phase D2** (no fabricated stats — every figure sourced or ranged/dated; case-study magnitudes real).
- Embed **`.webp` only**: `/imgs/blogs/<slug>-<n>.webp` (≥7 figures/post). Verify gate: `bash .claude/skills/finance-writer/scripts/verify-finance-post.sh <post.md> <slug> deep-dive`.
- Per wave: commit **only that wave's** `.md` + `.webp` with explicit paths (never `git add -A`) → push to `main`.
- Clean `.cache/finance-writer/<slug>/` after each green post.
- `## Sources & further reading` on every post (sourcing gate).
- LaTeX gotchas: brace-wrap digit-leading inline math (`${0.9}$`); use `\lt ` for `<`+letter inside math.

## Cross-link targets (existing posts)
- `trading/finance/enron-2001-accounting-fraud.md` · `trading/finance/madoff-ponzi-scheme.md` · `trading/finance/lehman-brothers-2008-financial-crisis.md` · `trading/finance/inside-an-investment-bank-how-they-make-money.md`
- `trading/equity-research/**` (valuation, quality-of-earnings posts) · `trading/analyst-edge/**` (thesis, red flags) · `trading/asset-valuation/**`
- `trading/vietnam-stocks/**` (VN context for Wave 8)

---

## WAVE 1 — Foundations: How to Read the Statements (5)
1. **the-three-financial-statements-and-how-they-interlock** — The master map: income statement, balance sheet, cash flow — what each measures and how a single transaction flows through all three. Why net income and cash diverge. The frame the whole series builds on. Case: how one fake sale ripples across all three statements.
2. **reading-the-income-statement-and-the-quality-of-earnings** — Revenue → COGS → operating income → net income; recurring vs one-off; the concept of *earnings quality* (are profits backed by cash?). Gross/operating/net margin and what moves them. Worked P&L.
3. **reading-the-balance-sheet-what-companies-hide-here** — Assets = liabilities + equity; current vs non-current; the accounts that flatter (receivables, inventory, goodwill, capitalized costs) and the accounts that hide (contingencies, off-BS items). Solvency vs liquidity.
4. **reading-the-cash-flow-statement-why-cash-beats-net-income** — CFO/CFI/CFF; the indirect method; why cash from operations is the hardest number to fake and the first place to look; free cash flow. Reconciling net income to CFO line by line.
5. **accrual-accounting-versus-cash-the-gap-fraud-exploits** — Accruals: the legitimate reason earnings ≠ cash, and the exact seam every earnings manipulation lives in. Matching principle, revenue recognition timing, the accrual reversal problem. Sets up the whole toolkit.

## WAVE 2 — Reading Between the Lines (5)
6. **the-footnotes-and-mda-where-the-bodies-are-buried** — Why the notes and Management Discussion & Analysis often matter more than the face statements: accounting-policy changes, segment data, related parties, contingencies, subsequent events. How to speed-read a 10-K/annual report for red flags.
7. **non-gaap-and-adjusted-ebitda-the-metrics-companies-invent** — The gap between GAAP net income and "adjusted"/"community-adjusted" EBITDA; which add-backs are legitimate and which are abuse; how recurring costs get relabeled one-off. Case: WeWork's "Community Adjusted EBITDA."
8. **common-size-and-trend-analysis-making-statements-comparable** — Normalizing statements to percentages and time series to spot the anomaly: a margin that drifts, receivables growing faster than sales, an expense line that vanishes. The analyst's first-pass screen.
9. **the-cash-conversion-cycle-and-what-working-capital-reveals** — DSO, DIO, DPO and the cash conversion cycle as an early-warning system; how channel stuffing and receivables games show up here first. Worked cycle for a healthy vs a stressed firm.
10. **how-an-audit-works-and-what-it-does-not-catch** — What auditors actually test, materiality, the audit opinion types, going-concern, and the structural limits (management override, collusion, sampling) that let big frauds slip through. Sets up the audit-red-flag detection post.

## WAVE 3 — The Manipulator's Toolkit I: The Income Statement (5)
11. **revenue-recognition-games-channel-stuffing-and-bill-and-hold** — Pulling tomorrow's sales into today: channel stuffing, bill-and-hold, premature recognition, percentage-of-completion abuse. The purpose (hit the quarter) and the tell (receivables balloon). Case: Sunbeam / bill-and-hold.
12. **round-tripping-and-fabricated-revenue** — Fake sales with no economic substance: round-trip/wash trades, related-party revenue, phantom customers. Why it inflates the top line with no cash. Case: Enron/dot-com round-trips, Luckin preview.
13. **capitalizing-costs-to-inflate-profit-the-worldcom-move** — Moving expenses off the income statement onto the balance sheet as assets; capitalizing what should be expensed; the software/R&D grey zone. The mechanical profit boost. Case: WorldCom line costs.
14. **cookie-jar-reserves-and-big-bath-accounting** — Over-reserving in good years to release ("smooth") in bad ones; the big-bath write-off that resets the bar. How reserves become an earnings dial. Case: classic SEC "cookie jar" enforcement.
15. **inventory-and-receivables-inflation-the-classic-red-flag** — Overstating inventory (fewer write-downs → higher profit) and receivables (booking uncollectible sales); why these two accounts are fraud's favorite home. The gross-margin and DSO tells.

## WAVE 4 — The Manipulator's Toolkit II: The Balance Sheet & Structure (5)
16. **off-balance-sheet-financing-and-special-purpose-entities** — Moving debt and losses into entities that don't consolidate; SPEs/VIEs, the equity-at-risk rules, synthetic leases. The purpose (hide leverage, book gains). Case: Enron's Raptors/LJM.
17. **related-party-transactions-and-self-dealing** — Deals with insiders, affiliates, and controlled entities that move value out or fabricate it in; why they're the single most reliable fraud marker. How to find them in the notes. Case: Adelphia / Rigas family.
18. **goodwill-intangibles-and-the-impairment-timing-game** — How acquisitions create goodwill, and how delaying or front-loading impairment massages earnings; purchase-price-allocation games. The write-down that never comes.
19. **hidden-liabilities-leases-guarantees-and-contingencies** — Operating-lease-era hidden obligations (and the ASC 842/IFRS 16 fix), guarantees, litigation reserves, and pension underfunding kept off the face statements. What to reconstruct.
20. **shell-companies-reverse-mergers-and-how-fraud-gets-listed** — How a fraud reaches public markets cheaply: reverse mergers, SPAC shells, chains of holding companies, and why the structure itself is a warning. Case: the China reverse-merger wave.

## WAVE 5 — Cash-Flow & Advanced Manipulation (5)
21. **cash-flow-statement-manipulation-classification-shifting** — Even "hard" CFO can be gamed: moving outflows from operating to investing/financing, misclassifying, capitalizing operating costs, stretching payables at quarter-end. The tricks that flatter free cash flow.
22. **factoring-supplier-financing-and-hiding-debt-in-plain-sight** — Receivables factoring, reverse factoring / supply-chain finance, and vendor financing used to window-dress leverage and working capital. Case: Carillion / Greensill dynamics.
23. **pension-deferred-tax-and-the-estimate-based-accounts** — Discount-rate and return assumptions on pensions, deferred-tax-asset valuation allowances — the "estimate" accounts management can dial. How small assumption tweaks move reported profit.
24. **stock-based-compensation-buybacks-and-eps-optics** — How SBC hides real cost from adjusted metrics, and how buybacks flatter EPS while dilution offsets it; the difference between per-share optics and value creation.
25. **transfer-pricing-and-offshore-profit-shifting** — Moving profit across borders via intra-group pricing, IP holding companies, and tax havens; legitimate vs abusive; why the geography of profit is a red flag. Case: the tech-giant structures (Double Irish).

## WAVE 6 — The Detection Toolkit (5)
26. **the-beneish-m-score-detecting-earnings-manipulation** — The 8-variable model that flagged Enron; each ratio explained (DSRI, GMI, AQI, SGI, DEPI, SGAI, LVGI, TATA), the −1.78 threshold, and a full worked calculation. Strengths and false positives.
27. **the-altman-z-score-predicting-financial-distress** — The bankruptcy predictor; the five ratios, the zones (distress/grey/safe), and how a deteriorating Z-score front-runs a blow-up. Worked calc; where it breaks (banks, asset-light firms).
28. **the-accruals-ratio-and-the-accruals-anomaly** — Sloan's insight: high-accrual earnings mean-revert and underperform; computing balance-sheet and cash-flow accruals; using the accruals ratio to rank earnings quality. Worked example.
29. **forensic-ratios-dso-dio-dpo-and-margin-anomalies** — The ratio dashboard that catches manipulation: receivables/sales, inventory/COGS, margins vs peers, revenue-vs-CFO divergence, the "cash-realization" ratio. Building a red-flag scorecard.
30. **benfords-law-and-digit-analysis-for-fraud** — Why real financial figures follow Benford's leading-digit distribution and fabricated ones don't; how forensic accountants and regulators use digit tests. Worked test on a fabricated vs real dataset.

## WAVE 7 — Detection in Practice + Case Studies I (5)
31. **red-flags-in-the-audit-report-and-auditor-changes** — Reading the audit opinion, critical audit matters, going-concern language, restatements, and the biggest tell of all: an unexplained auditor resignation or a switch to a small firm. Case: Wirecard/EY, and the auditor-shopping pattern.
32. **the-short-sellers-playbook-how-activists-find-fraud** — How Muddy Waters, Hindenburg, Citron and others actually build a fraud thesis: ground-truthing revenue, satellite/traffic checks, related-party mapping, and publishing. The research process, reproduced. Case: the anatomy of a short report.
33. **enron-a-forensic-re-read-of-spes-and-mark-to-market** — Re-reading Enron through this series' toolkit: mark-to-market revenue, the SPE web, what the statements showed vs hid, and which red flags fired first. (Cross-links the existing Enron post.)
34. **worldcom-the-11-billion-dollar-capitalization-fraud** — The largest accounting fraud of its era, mechanically: how capitalizing line costs turned losses into profits, how internal audit caught it, and how the cash flow statement would have exposed it.
35. **wirecard-the-missing-1-9-billion-euros** — Fabricated third-party-acquirer cash and escrow balances that never existed; how a DAX-30 darling faked an entire cash pile, why the audit missed it for years, and how short-sellers and the FT got there first.

## WAVE 8 — Case Studies II + Emerging Markets (5)
36. **luckin-coffee-fabricated-sales-and-the-muddy-waters-report** — Inflated per-store sales and fabricated transactions at a US-listed Chinese chain; how the Muddy Waters/anonymous field-work exposed it, the statement tells, and the aftermath.
37. **satyam-and-toshiba-two-faces-of-asian-accounting-fraud** — India's "Enron" (fictitious cash and fake invoices, founder confession) vs Toshiba's culture-driven, top-down profit-inflation (percentage-of-completion). Two very different mechanisms, one lesson.
38. **lehman-repo-105-and-window-dressing-the-balance-sheet** — Using Repo 105 to move ~\$50bn of assets off the balance sheet at quarter-end to flatter leverage; legal-but-deceptive window dressing and what it teaches about period-end games. (Cross-links the Lehman post.)
39. **steinhoff-and-the-anatomy-of-a-global-accounting-collapse** — A sprawling multinational's fictitious/related-party transactions and off-balance-sheet structures; how complexity itself concealed the fraud, and how to read a group this tangled.
40. **reading-financial-statements-in-vietnam-red-flags-and-the-flc-case** — Applying the whole toolkit to an emerging market: VN reporting quirks, related-party and pledged-share risks, market manipulation vs accounting fraud, and the FLC / Trịnh Văn Quyết case. Cross-links VN-stocks posts. The series capstone.

---

## Execution log
- **2026-07-14** — Series planned, folder created, memory written. Language=English, scope=full 40, folder=trading/forensic-accounting. Wave 1 next.
- **2026-07-17** — Post 5 (accrual-vs-cash) shipped, commit `3b2507b1`. Salvaged from a dying agent: it had written the .md and rendered all 8 PNGs but died before the WebP step.
- **2026-08-03** — **WAVE 1 COMPLETE (5/5)**, commit `e83ba127`. Posts 1–4 shipped: three-statements (10,733w/8 figs), income-statement (11,638w/9 figs), balance-sheet (13,366w/9 figs), cash-flow (11,627w/9 figs). All pass the deep-dive gate; every figure passed the Phase C2 visual review.
  - **Renderer path gotcha (cost this wave ~25 min).** All four agents authored scene JSON and rendered nothing. `SKILL.md` cites `mcp_excalidraw/scripts/render-scene-batch.mjs`, which does not exist relative to the repo root. The real path is **`/Users/hieptran1812/Documents/mcp_excalidraw/scripts/render-scene-batch.mjs`** — a sibling directory, outside the repo. Pass this absolute path to every figure agent. It logs a harmless 404 and a `[page error] ...reading 'length'` while still writing a correct PNG.
  - **The verify gate passes vacuously when no figures exist on disk.** `verify-post.sh` sharpness loop is `for f in …-[0-9]*.webp; do [ -e "$f" ] || continue`, so zero matches means zero iterations and a PASS; diagram-count only counts `![` embeds in the markdown. A post with every image broken reports "all gates passed". **Always confirm the WebP count on disk before believing a green gate.**
  - **Sparse line charts fail the 40 KB byte floor.** Two-line net-income-vs-CFO charts came out at 23–29 KB because a couple of thin polylines on a wide canvas compress to nothing losslessly. Fix by adding real ink — end-of-line labels, per-point values, axis ticks, a shaded gap band, tighter canvas — not by inflating the file.
- Sourcing WARN fires on 106–124 lines per post (illustrative worked-example numbers). Expected for this series; not blocking.
- **Visual-review memory (2026-08-04, Wave 2):** User caught a broken early-warning dashboard figure (cash-conversion-cycle post, figure 8): long arrows converged into one point under the center and labels were cramped. For every future figure, inspect the rendered WebP for arrow crossings/convergence, legible numeric spacing/decimals, and readable hierarchy; do not accept a green automated gate as visual approval.
- **2026-08-04** — **WAVE 2 COMPLETE (5/5)**, commit `b75c3246`. Posts 6–10 shipped: footnotes/MD&A (8,503w/8 figs), non-GAAP/adjusted EBITDA (8,598w/7 figs), common-size/trend analysis (6,159w/7 figs), cash-conversion cycle (6,046w/9 figs), and audit limits (8,413w/9 figs). All pass the deep-dive gate and blog validation. Figure 8 of the cash-conversion post was redesigned after user visual review to remove converging arrow clutter; sourcing WARN remains for explicitly illustrative worked-example numbers. The next wave is Wave 3 — The Manipulator's Toolkit I: The Income Statement.
- **2026-08-04** — **WAVE 3 COMPLETE (5/5)**, commit `8d0604f3`. Posts 11–15 shipped: revenue-recognition games (8,558w/7 figs), round-tripping/fabricated revenue (8,072w/8 figs), capitalizing costs/WorldCom (8,757w/7 figs), cookie-jar reserves/big baths (6,128w/7 figs), and inventory/receivables inflation (6,548w/7 figs). All pass the deep-dive gate and blog validation; 36 WebPs total. Round-tripping required a post-render gate fix: its cash-conversion figure was moved next to the explanatory bridge so abstraction coverage passed. Sourcing WARN remains only for clearly labeled illustrative arithmetic. Visual checks used scene validation, WebP decode/dimensions/size checks, and figure-specific layout requirements; no separate figure-reviewer tool was available to the agents.
- **2026-08-04** — Follow-up fix commit `abc10f46`: round-tripping received the final bridge-coverage placement and two re-rendered WebPs after the original wave commit; verifier and blog validation remained green.
- **2026-08-04** — **WAVE 4 COMPLETE (5/5)**, commit `294dec32`. Posts 16–20 shipped: off-balance-sheet financing/SPEs (6,191w/7 figs), related-party transactions (8,631w/9 figs after removing duplicate embeds), goodwill/intangibles impairment timing (6,108w/7 figs), hidden liabilities/leases/guarantees (6,534w/7 figs), and shell companies/reverse mergers (8,533w/8 figs). All finance gates and blog validation passed; 38 Wave 4 WebPs shipped. Sourcing WARN remains only for clearly labeled illustrative arithmetic. Also fixed the user-reported `MATH_BLOCK_2` placeholder in the Wave 3 capitalizing-costs figure 3 (2060×1469 WebP, placeholder scan clean). Specialized figure-reviewer/post-verifier subagents were unavailable, so local validation and disk-level image checks were used.

- **2026-08-05** — Wave 5 partial, commits `39a80e89` (cash-flow classification-shifting 13,339w/8 figs; SBC & buyback optics 12,960w/9 figs) and `baa75344` (transfer pricing & profit shifting 13,057w/9 figs).
- **2026-08-09** — **WAVE 5 COMPLETE (5/5)**, closing commit `39b060b6`. The last two posts had been drafted and rendered but never committed: factoring/supplier-financing (10,747w/8 figs) and pension/deferred-tax estimate accounts (10,939w/9 figs). Both now pass the deep-dive gate, as does the re-gated SBC post.
  - **Two gate failures found on the stranded posts, both fixed.** (a) `factoring-…-5.webp` was a sparse 2648×1304 at 27 KB, under the 40 KB sharpness floor. Root cause was **not** the figure design — the dense redesign (gridlines, ticks, per-point labels, shaded 89-day gap band, legend) already existed in `.cache/…-5.in.json` but had **never been run through the pipeline**: it was a `"type": "raw"` DSL input routed through `layout-scene.mjs`, not `author-scene.mjs`. Regenerating the scene from it and re-rendering gave 1911×1304 / 52 KB, reviewer PASS. **Check `.cache` for an unrendered redesign before re-authoring a thin figure.** (b) pension post failed `abstraction-coverage`; the nearest embeds were 83 and 104 lines away and too tightly bound to their own sections to move honestly, so the fix grounded the claim in the post's own \$1.5 bn Northfield deficit instead of relocating a figure.
  - `figure-reviewer` and `post-verifier` subagents **were** available this session (Wave 3/4 logged them as unavailable) and worked as intended.
  - Push was briefly blocked: **github.com was unreachable over both HTTPS and SSH** for ~10 minutes while the rest of the internet was fine (DNS resolved; Google/Wikipedia/sec.gov all responded). Waited it out and pushed cleanly — `f0970b99..39b060b6` is on `origin/main`.

- **2026-08-11** — **WAVE 6 COMPLETE (5/5)**, commits `75ae8d56` (Benford), `53d58ffc`+`aaa45a07`+`01b1bc16` (accruals), `089d8bef`+`8c3333a1` (forensic ratios), `2ffa6a99` (Beneish), `1fa0b049` (Altman). Plus `e5ca16f2`, a correction to a Wave-1 post. Posts 26–30: Beneish M-Score (10,904w/10 figs), Altman Z-Score (11,090w/9), accruals ratio (12,450w/8), forensic ratios (11,886w/9), Benford's Law (11,765w/8). 44 WebPs. Series now **30/40**.
  - **Ran over ~40 hours across two session-limit resets and a GitHub outage.** Both limit hits killed every agent; **resuming via `SendMessage` recovered all of it** — four of five drafts survived the first hit on disk. Re-dispatching would have paid for the research twice and is what creates duplicate writers.
  - **Sourcing correction that propagated into a shipped post.** The M-Score threshold −2.22 does **not** appear in Beneish (1999); its eight-variable cutoffs are −1.78 (20:1/30:1 error cost) and −1.49 (10:1). −2.22 belongs to Beneish's earlier **five-variable** model (1997, *JAPP* 16(3), 271–309) — different intercept, non-comparable scale. Wave-1's `accrual-accounting-versus-cash-the-gap-fraud-exploits` had attributed it to the eight-variable score; fixed in `e5ca16f2`. Secondary sources also call −2.22 "stricter", which is backwards (the rule flags scores *above* the cutoff). The accruals post separately corrected a Fama-French/asset-growth conflation and an author-name error.
  - **Two of the orchestrator's own confident diagnoses were wrong, and agents caught both by verifying.** (a) I called Beneish figs 3↔7 and 9↔10 transposed and told the agent to fix the render manifest — the manifest was a clean identity map and all four figures already held their specced content; the **reviewers had been handed each other's specs**. Acting on my instruction would have corrupted four correct charts. (b) I told the Benford agent its log-ruler ticks were evenly spaced — it measured the PNG (positions matched log₁₀(d) to within 0.003) and correctly left the file alone. **Instruct agents to verify before destructive fixes.**
  - **Do not commit while a figure agent is still iterating.** The forensic-ratios post was committed on its drafter's "done" while `ratio-figures-3` was mid-fix; six figures were superseded minutes later and needed a second commit.
  - **Cap figure fix-loops.** Figure 8 of forensic-ratios took six rounds in one context and each fix exposed the next defect. What ended it was handing it to a **fresh** agent with the diagnosis — it passed on the first cycle. Cap at 2–3 rounds, then re-dispatch clean.
  - **Accepted defect:** forensic-ratios fig 5's bars fill ~70% of frame width with wide gutters. Reviewer confirms data/labels/colours/reading-order all correct. Cosmetic; shipped.
  - New reusable bugs recorded in memory: **bound text keeps its own `y`** (moving a container orphans its label, rendering an empty box); **half-bound text renders invisible**; **literal `$` must be escaped `\$`** (~140 in one draft would have desynced the math parser); **`timeout` does not exist on macOS** and mimics a render failure; **WebFetch of sec.gov returns 403** — use `WebSearch` with `allowed_domains: ["sec.gov"]`.

## DONE: Wave 8 shipped 2026-08-12 — series complete at 40/40 (was: Next)
Start a **fresh session** — and for this one, **restart the app rather than `/clear`**: the WebSearch and
subagent caps are per-CLI-process, and Wave 7 exhausted the search budget mid-wave with five drafters each
spawning a research child. Carry forward only this plan path and the wave number.

**Wave 8 carries six posts: the five planned (36–40) plus deferred post 32,
`the-short-sellers-playbook-how-activists-find-fraud`. Dispatch 32 FIRST** — it is the most research-hungry
post in the series and needs the freshest search budget. Its currency notes are in the Wave 7 log above.

**FACT-CHECK CRITICAL, and more so than Wave 7** — Luckin, Satyam, Toshiba, Lehman, Steinhoff and FLC involve
living people, ongoing matters and, in the FLC case, a jurisdiction where the enforcement record is harder to
source in English. Every figure sourced or dated-and-attributed; contested claims framed as reported/alleged
with the source named; anything unverifiable dropped rather than softened. Watch research children for
Wikipedia-sourced case studies presented as primary — that pattern produced a real error in Wave 7.

- **2026-08-11** — **WAVE 7: 4 of 5 shipped.** Commits `4334b9b9` (WorldCom, 11,663w/8 figs), `6100b2ed`
  (Enron, 11,103w/9), `5f53cf73` (Wirecard, 12,138w/9), `d36a45a2` (audit red flags, 12,779w/10). **36 WebPs,
  ~47,700 words.** Series now **34/40**. **Post 32 `the-short-sellers-playbook-how-activists-find-fraud` was
  deferred to Wave 8** — see below.
  - **Post 32 deferred, deliberately.** Its drafting agent gave a straight answer when asked: it held **zero
    sourced case facts** (both research children died before returning anything) and had written no prose. It
    is the most research-hungry post in the series and drew the worst hand — its fact-check child died twice
    and the WebSearch budget was gone. It is also the one post where under-sourcing could genuinely harm
    someone, since it names living short-sellers and the companies they accused. **Re-dispatch it first in
    Wave 8, in a fresh process with a full WebSearch budget.** Currency notes for whoever takes it:
    **Hindenburg Research wound down in January 2025** (past tense, or the post reads stale on day one), the
    **Andrew Left / Citron matter is charges against a living person** needing an explicit as-of date, and
    Rota Fortunae/Farmland is the counter-case item that can be stated plainly because the retraction and
    settlement are matters of record.
  - **Three environment failures, in one wave.** Two session-limit wipes (the second landing three minutes
    after the first reset) and a DNS outage (`ENOTFOUND`) that killed five agents at once. **Resuming by name
    via `SendMessage` recovered everything**; two finished drafts and six gated figures were sitting on disk
    when the drafters died. Re-dispatching would have paid twice and clobbered them.
  - **The `figure-author` → fresh-agent handoff is now measured three times.** Wirecard figs 2 and 6 and audit
    fig 9 each failed **two** in-context rounds where every fix exposed the next defect, then passed on the
    **first** cycle with a fresh agent holding the written diagnosis. Cap at 2 rounds and hand off — do not
    negotiate with a figure.
  - **Two timeline figures had non-monotonic or unanchored axes** (Wirecard 6: "Sep 2018" plotted left of
    "Feb 2016"; audit 9: four of five cards floating with no axis line). Both would have shipped green — the
    automated gate counts files and checks pixels and bytes, and **cannot see that a timeline runs backwards**.
    Only the visual reviewer catches this class. It is a faithfulness failure, not a cosmetic one.
  - **Orchestrator error worth not repeating:** I read a figure's unchanged mtime as "the agent never started",
    announced it, and dispatched a duplicate that nearly overwrote a **passing** figure. The child had been
    running eight minutes and simply had not written yet. Wave 6 recorded the mirror image (a confident
    misdiagnosis that nearly corrupted four correct charts). **Absence of a file is not evidence of an idle
    agent.** Aborted in time; nothing lost.
  - **Two gate FAILs are documented deviations, NOT defects — do not "fix" them:**
    - `em-dash` fails **series-wide**; shipped Wave-6 post `benfords-law-…` fails it with **107 lines**.
      Stripping them from one post would break style consistency with its 34 siblings.
    - `worked-examples: 0 $-figures` on **Wirecard only** — the check counts `\$[0-9]` and this is the series'
      first **euro-denominated** post. It has five step-by-step money walkthroughs the regex cannot see.
      Converting a German company's figures to dollars to satisfy a US-centric lint would be worse for the reader.
  - **Fixed by hand in-session:** WorldCom shipped with 17 ` ```text ` blocks failing `forbidden: text fenced
    'diagrams'`. They were formulas, journal entries and statement fragments, not ASCII art — retagged by
    content to the house convention (2 → ` ```journal `, 15 → bare fences). `​```text` now appears nowhere in
    the series.
  - **Research-child sourcing hazard.** Fact-check children returned **primary** sources for regulation (PCAOB,
    IAASB, FASB, 17 CFR, SEC) but **Wikipedia** for case studies, in two places with en/de contradictions.
    Caught before shipping: **Braun's trial began 8 December 2022, not December 2024, and no verdict is
    verifiable** — the Wirecard post asserts none. Also corrected: CAM is a gateway + **two** prongs, not
    three; KPMG's special-audit report is **28 April 2020**; AS 2415's going-concern window is one year beyond
    the **financial-statement date**, a different clock from ASC 205-40's one year after **issuance**.
  - **WebSearch 200-cap is per-CLI-process** and one research child exhausted it for all five drafters
    mid-wave. `WebFetch`/`curl` are uncapped — `curl` on
    `sec.gov/Archives/edgar/data/<cik>/<accession>.txt` with a descriptive UA works despite WebFetch 403ing.

**Enron's thesis, preserved for cross-linking.** Computed from originally-filed lines: the
accruals ratio (**−7.7%** in 2000) and Altman **Z = 2.45** (grey) *do not fire*; what fires is **Z'' = 0.80**,
ROA 3.62% → **1.49%**, assets **+96%** vs net income **+9.6%**, and **Note 16 of the FY2000 10-K**, which
disclosed the \$1.2 bn of contributed assets, \$172.6 m of Entity cash in Enron demand notes and ~\$500 m of
derivative revenue *in the published filing*. Also: FY1999 10-K names "LJM Cayman, L.P." / "LJM2
Co-Investment, L.P."; the FY2000 10-K replaces both with "the Related Party" — a dated two-filing tell.
EDGAR accessions: FY1996 `0000072859-97-000009` (**CIK 72859**, not 1024401 — that one is an all-zero merger
shell), FY1997 `0001024401-98-000009`, FY1998 `0001024401-99-000007`, FY1999 `0001024401-00-000002`,
FY2000 `0001024401-01-500010`, restatement 8-K `0000950129-01-503835`.

**Two gate FAILs on this wave are NOT defects — do not "fix" them:**
- `em-dash` — fails **series-wide**. Shipped Wave-6 post `benfords-law-…` fails it with **107 lines** and
  `RESULT: gates FAILED`. Standing accepted deviation; stripping them in one post would break style
  consistency with its 34 siblings.
- `worked-examples: 0 $-figures` on the **Wirecard** post only — the check counts `\$[0-9]`, and this is the
  series' first **euro-denominated** post. It has five step-by-step money walkthroughs; the regex cannot see
  euros. Keeping EUR is correct for a German company; converting to mixed currency to satisfy a US-centric
  lint would be worse for the reader. Accepted deviation.

**Wave-7 process lessons:** the **WebSearch 200-cap is per-CLI-process** and one research child exhausted it
for all five drafters mid-wave — `WebFetch`/`curl` are uncapped, use those (curl on
`sec.gov/Archives/edgar/data/<cik>/<accession>.txt` with a descriptive UA works, despite WebFetch 403ing).
Research children returned **Wikipedia-sourced case studies** with en/de contradictions; caught a real error
before it shipped — **Braun's trial began 8 December 2022, not December 2024, and no verdict is verifiable**.
Also corrected: CAM is a gateway + **two** prongs; KPMG's special-audit report is **28 April 2020**; AS 2415's
going-concern window is one year beyond the **financial-statement date**, a different clock from ASC 205-40's
one year after **issuance**.

**Wave 6 dispatch note (2026-08-09):** all five agents were dispatched, then four died within ~15 min on transient network errors (`SSL certificate verification failed`, `Connection closed mid-response`) during the same window that took GitHub offline. **Nothing reached disk** — no `.md`, no WebPs, one stray cache file. They were resumed in place via `SendMessage` rather than re-dispatched, with instructions to re-check disk state first, since orphaned `figure-author` children of the dead parents were still running and a second figure pass would clobber. If a wave dies this way again: resume, do not re-dispatch, and always have the resumed agent inventory the disk before authoring figures.

- **2026-08-12** — **WAVE 8 COMPLETE (6/6). SERIES COMPLETE, 40/40.** Commits `744d9d5a` (Luckin
  11,853w/9 figs + Steinhoff 14,692w/8), `461ea9bc` (Luckin final renders), `3b066c25` (Satyam+Toshiba
  14,086w/10), `d7d26590` (Lehman Repo 105 12,536w/8), `e41c67eb`+`4d1f50c5` (short-sellers' playbook
  13,626w/9), `a5368493` (Vietnam/FLC capstone 14,018w/10). **54 WebPs, ~80,800 words.**
  Wave 8 carried six posts: the five planned plus post 32 deferred from Wave 7.

  - **Two session-limit hits inside ~15 minutes killed all six drafters and every child, twice.**
    Resuming via `SendMessage` recovered everything both times; nothing was re-dispatched and no research
    was paid for twice. Surviving-on-disk state after hit one: 4 drafts, 29 WebPs. The rule holds and is
    now proven three waves running.
  - **The visual gate had never run on 9 short-sellers figures.** Its figure-author died right after
    "all nine clear the sharpness floor, dispatching the visual gate". The sharpness floor is a **byte-size
    check, not a visual review**: re-gating found **3 of 9 failing** (orphaned edge labels, empty band above
    bars, payoff line running off-axis). One had a real content bug, figure 5's purchases arrow pointing
    *into* the issuer while its label read "out". **A figure that only cleared the byte floor is ungated.**
  - **I reported a dollar-escaping defect that did not exist**, on two posts, and retracted it before either
    agent acted. My grep was not fence-aware: all 18 satyam and 12 of 13 steinhoff `$[0-9]` hits were
    **inside code fences** where `$` is literal, and the one outside was correct display math with escaped
    `\$`. Escaping them would have rendered visible backslashes. **Fence-aware scan:**
    `awk '/^```/{f=!f;next}{l=$0;gsub(/\\\$/,"",l); if(l~/\$[0-9]/) print (f?"fenced":"BARE"), NR}'`
  - **I committed twice while a drafting/figure agent was still working** (Luckin, then short-sellers), each
    needing a follow-up commit. Wave 6 recorded this exact lesson and I repeated it. **A green post-verifier
    does not mean the agent has stopped.** Check `ListAgents` for running children *and* compare file mtimes
    against now before staging. Banking work early against limit hits is still right; the precondition is an
    idle agent, not a green gate.
  - **Small batches beat one big figure pass.** Vietnam lost two entire sessions to figure-authors that read
    `layout-scene.mjs`/`author-scene.mjs` end to end and never reached the renderer. Re-dispatched as
    batches of 2-3 with the validator findings handed to them, all 10 landed. **Author, run the validator,
    react** rather than studying the engines first.
  - **New reusable renderer findings:** validator rule 3 skips `arrow` but **not** `line`, and rule 3c checks
    arrows against rectangles, so **a curve cannot cross a shaded rectangular region** (band = closed 5-point
    `line` polygon with `fillStyle:"solid"`, curve = `arrow`, axes = zero-width `line`; in-band text needs a
    cutout or must sit outside the bbox). `author-scene.mjs` **does not read the `raw.elements` wrapper**
    (reports 0% coverage / 0 tokens) so `raw` DSL must go through `layout-scene.mjs`. The `grid` engine caps
    row growth at `bodyH*0.5`, so a 7-row grid with 2-line cells fails containment. `raw` bound labels are
    never auto-wrapped. Body font floor is 22.
  - **`abstraction-coverage` fired on 2 of 6 posts.** Correct fix is to ground the passage in the post's own
    arithmetic, not to drag a figure away from the argument it illustrates. Short-sellers closed it by adding
    the two-division extrapolation (20 counted in an hour at a 12-hour branch implies ~240/day against a
    filing's implied 150) to the "go stand somewhere and count" paragraph, which improved the post.
  - **Every post shipped with a stale `readTime`** (declared 48-61 against recomputed 57-67). Worth setting
    from the verifier's recompute as a standing final step.
  - **Fact-check outcomes.** WebSearch hit its 200/process cap early and every alternative engine was
    captcha-walled; `WebFetch`/`curl` carried the wave. Unverifiable claims were **dropped, not softened**:
    the Citron/Andrew Left enforcement matter is absent from the short-sellers post entirely; Steinhoff ships
    an explicit "claims deliberately left out" section (FY2016 revenue/profit, goodwill share, delisting
    dates, settlement pot) after web.archive.org 429'd. Steinhoff's "on 5 December 2017 the board announced"
    was **caught as unsourced before shipping** and reframed to the Mail & Guardian's 6 December report with
    a note on the unsettled date. Lehman states the Examiner's **colourable claims** finding rather than
    implying adjudicated fraud, and figure 7 marks a data gap "not tabulated" instead of interpolating.
    Vietnam keeps FTSE **announced and confirmed, not yet effective** (first tranche 21 Sep 2026, full
    inclusion Sep 2027), MSCI still Frontier, and separates the 5 Aug 2024 first-instance verdict from the
    26 Jun 2025 appellate judgment.
  - Em-dash gate: **0 across all six posts**, the first wave written entirely under the no-em-dash house rule.
    Sourcing WARN (63-122 lines/post) remains the accepted series-wide pattern for illustrative arithmetic.
