# Fact-check protocol — real numbers, every number sourced

Read at the start of **Phase D2** (after the draft is written, before the verify gates). This is the finance-writer's non-negotiable data-integrity layer. A finance post lives or dies on its numbers; one fabricated figure poisons the reader's trust in the whole article.

## The one rule that overrides everything

> **Never invent a number.** If you cannot trace a figure to a real source, you do not get to write it. You have exactly three honest moves, in order of preference:
>
> 1. **Source it** — find the real value and cite it (with its as-of date).
> 2. **Range it** — if the exact value is unverifiable but the order of magnitude is solid, write a sourced range (`roughly $180–220 billion`) and say why it's a range.
> 3. **Date-and-attribute it** — if it's a moving number you can only pin to a moment, write the value *with* its timestamp and source (`the VN-Index closed at 1,245 on 2026-06-20 (HOSE)`), so the reader knows it's a snapshot, not a law of nature.
>
> What you may **never** do is state a precise-looking figure you made up, rounded from memory, or "estimated" without saying so. A confident wrong number is worse than an honest "exact figure unavailable; on the order of X."

Round, friendly numbers in **worked examples** (`you buy 1 share at $100`) are explicitly fine — they are illustrative arithmetic, not claims about the world, and the prose already frames them as hypothetical ("suppose…", "imagine you buy…"). The rule above governs **factual claims about reality**: market levels, rates, sizes, dates, regulatory limits, historical magnitudes.

## What counts as a "quantitative claim"

Any statement asserting a fact about the world that contains, or implies, a number:

- **Prices / levels** — a stock price, an index level, an FX rate, a yield, a spread, a premium.
- **Rates / ratios** — interest rates, tax rates, inflation, growth, ownership caps, margin limits, reserve ratios, P/E, default rates.
- **Sizes / flows** — market cap, AUM, trading volume, GDP, debt, money supply, trade balance, foreign flows.
- **Dates / durations** — when a law took effect, when an event happened, how long a cycle ran, settlement windows (T+2).
- **Counts** — number of listed firms, number of banks, headcount, defaults, basis-point moves.
- **Magnitudes of historical episodes** — "the index fell 40% in 2008", "the dong devalued 9% in 2015".

If a sentence would be falsifiable by checking a source, it is a quantitative claim and it goes in the ledger.

## The four-step process

### Step 1 — Extract every quantitative claim into a ledger

Run the extractor on the drafted post:

```bash
bash .claude/skills/finance-writer/scripts/extract-claims.sh content/blog/trading/<sub>/<slug>.md
```

It prints four blocks:

- **CLAIM LEDGER** — every line carrying a number, with its line number. This is your worklist: one row per claim.
- **NUMBER INDEX** — each distinct numeric token mapped to the lines it appears on (so you can eyeball whether a figure repeated across the post stays consistent).
- **POSSIBLE INTERNAL CONTRADICTIONS** — anchor terms (tickers, capitalized multi-word entities, a built-in finance-metric list) found near *two or more different numbers*. Each is a lead to check, not a verdict.
- **UNSOURCED LIVE-NUMBER LINES** — lines with a moving figure (price/index/rate/size) that have no nearby citation, link, or as-of date. These are your priority for Step 3.

Copy the CLAIM LEDGER into a working table (scratchpad, not the post) with columns: `claim | value-as-written | type | tier needed | source | as-of | status`.

### Step 2 — Grep for internal contradictions

Before reaching for any external source, make the post agree with *itself*. Internal contradiction is the cheapest error to catch and the most embarrassing to ship.

1. Walk the **POSSIBLE INTERNAL CONTRADICTIONS** block. For each anchor with multiple numbers, decide: are these the *same* quantity stated two different ways (a real contradiction — fix it), or two genuinely different quantities that happen to share a word (fine)? Example real bug: "the 10-year yield rose to 4.2%" in §2 and "the 10-year sits near 3.8%" in §5 with no time gap explained.
2. Walk the **NUMBER INDEX**. A figure you cite in the TL;DR, again in a section, and again in a figure caption must be byte-for-byte the same number (or explicitly reconciled — "≈$5.4tn" vs "$5,412bn" should be made consistent).
3. Check **unit drift**: bps vs %, million vs billion vs trillion, VND vs USD, nominal vs real, annual vs monthly. A number that is "right" in the wrong unit is a wrong number. Confirm every figure in a chart matches the worked example it illustrates (this is also a Phase C2 / Phase E check — do it here too).
4. Check **derived-number arithmetic**: if the prose says "revenue $120M, margin 25%, so profit $30M", recompute it. Worked examples must actually add up.

Resolve every internal conflict before external cross-check — there's no point sourcing a number you're about to change.

### Step 3 — Cross-check against external sources (the tier waterfall)

Each claim has a **native authoritative tier** — the kind of source that is *definitionally* correct for it. Route the claim to its native tier first; then, when tiers disagree, the higher tier wins. The four tiers, highest authority first:

| Tier | Source kind | Definitionally authoritative for | Examples |
|---|---|---|---|
| **1. Legal / primary text** | Laws, decrees, circulars, regulator filings & data, exchange rulebooks, central-bank releases, company filings | Regulatory numbers: tax rates, ownership/foreign caps, margin & reserve limits, decree numbers, effective dates, capital requirements, official policy rates, settlement cycles, a company's own reported financials | VN: *Công báo*/`vbpl.vn`, SBV (sbv.gov.vn), SSC/UBCKNN (ssc.gov.vn), MOF, GSO (gso.gov.vn), HOSE/HNX rulebooks, company prospectus/annual report. Global: SEC EDGAR, Federal Reserve, ECB, BIS, the statute itself |
| **2. Securities / financial press & market data** | Specialist financial/securities media and market-data venues | Live market data: prices, index levels, yields, FX, volumes, market cap, spreads, fund flows, earnings reactions | VN: CafeF, VietstockFinance, NDH, Đầu tư Chứng khoán, FiinGroup, exchange tickers. Global: Bloomberg, Refinitiv/Reuters markets, the exchange's own quote |
| **3. Mainstream / official press** | General-interest reputable outlets and official statistics agencies | General economic facts, event narratives, the GDP/CPI/unemployment prints, context around an episode | VN: VnExpress, Tuổi Trẻ, Thanh Niên, Vietnam News, GSO statistical releases. Global: AP, the national stats office, a finance ministry press release |
| **4. International cross-check** | Multilateral & global reference sources | Cross-validating a domestic figure, global aggregates, long-run series, comparisons across countries | IMF, World Bank, OECD, BIS, FT, The Economist, Reuters/Bloomberg wire |

**How to use the waterfall:**

- **Match first, then escalate.** A tax rate's home is Tier 1 (the decree) — don't cite a newspaper paraphrase of it when the decree is one search away. A stock's closing price's home is Tier 2 — don't go hunting for a law. Spend effort where the claim actually lives.
- **Two independent sources for anything load-bearing.** Any number in the TL;DR, a section thesis, or a headline figure gets corroborated by a *second, independent* source — ideally one tier up or a peer at the same tier. One source is a lead; two agreeing is a fact.
- **Higher tier breaks ties.** When the securities press and the decree disagree on an ownership cap, the decree is right and the press is stale or sloppy — cite the decree and, if useful, note the discrepancy. When international and domestic aggregates differ (common for GDP/debt due to methodology), say so and cite both rather than silently picking one.
- **Prefer primary over paraphrase, recent over old, the issuer over the commentator.** The SBV's own release beats a summary of it; the 10-K beats an article about the 10-K.
- **Searches are blocked? Say so.** If `WebSearch`/`WebFetch` is unavailable in this environment (it often is for batch/cron runs), you cannot invent the verification. Either (a) use a value you carried in from a cited data kit (`.cache/finance-writer/_<series>/data*.py` curated, cited series) and attribute it, or (b) apply the range/as-of fallback, or (c) leave a `<!-- TODO verify: … -->` note and flag it in the final report. Do **not** paper over an unverifiable number with false precision.

### Step 4 — Resolve and annotate

For each ledger row, the exit state is one of:

- **Verified** — value confirmed, source captured. Make sure the post lets the reader see the provenance (inline as-of date and/or a Sources entry; see conventions below).
- **Corrected** — value was wrong; fix the prose *and* any figure that repeated it, then re-run Step 1's extractor to confirm the fix didn't introduce a new contradiction.
- **Ranged** — exact value unverifiable; rewrite as a sourced range with the reason.
- **Dated** — moving value pinned to a moment; rewrite with the timestamp + source.
- **Cut** — couldn't source it, couldn't honestly range it, isn't load-bearing → delete the claim. A post is better with one fewer number than with one made-up number.

Update the working table's `status` column. The phase is done when no row is left in an unverified state.

## Sourcing & annotation conventions (how provenance shows up in the post)

The reader must be able to *check your work*. Two mechanisms, used together:

1. **Inline as-of date for live numbers.** Any moving figure carries its moment: *"the 10-year yield was 4.2% (as of 2026-06-20)"*, *"foreign ownership of the bank is capped at 30% under Decree 01/2014/NĐ-CP"*. The decree number, the date, the exchange — that *is* the citation for a regulatory or market fact and often suffices inline.
2. **A Sources section for load-bearing and non-obvious claims.** Deep-dive posts end with a short `## Sources & further reading` (it doubles as the existing "further reading" closer): a bulleted list of the primary sources behind the headline numbers — link text + publisher + as-of date. Inline, reference them naturally (a markdown link on the figure or term) rather than academic `[12]` superscripts, which read poorly in the blog's style.

You do not need a citation on every illustrative `$100`-share arithmetic step (those are hypotheticals). You **do** need provenance on every real-world figure: market levels, official rates, sizes, regulatory limits, dated historical magnitudes, company financials.

## When sources genuinely conflict

This is normal, especially for macro and cross-country data. Be honest in the prose:

- Pick the higher-tier / more-primary value as the figure you lead with.
- If the gap is material and interesting, *show* it: "estimates range from $180bn (World Bank, 2024) to $210bn (national statistics office, 2025) depending on whether informal activity is counted." That's not weakness — it's the kind of precision that earns a practitioner's trust.
- Never average two sources into a third number that no source reports. That manufactures a fake figure.

## Worked example of resolving one claim

> Draft sentence: *"Vietnam's foreign-ownership cap on banks is 30%, and CafeF reported VN-Index hit 1,300 last month."*

1. **Extract** → two claims: (a) 30% bank FOL cap [regulatory → Tier 1], (b) VN-Index 1,300 "last month" [market → Tier 2; also a *relative date*, a smell].
2. **Internal check** → "last month" is unanchored; if 1,300 appears again elsewhere as 1,245, that's a contradiction. Resolve to one absolute number.
3. **External** → (a) route to Tier 1: the aggregate foreign cap on a Vietnamese bank is 30% under Decree 01/2014/NĐ-CP — confirm the decree still governs and isn't superseded; the CafeF mention is a Tier-2 paraphrase, so cite the decree, not CafeF. (b) Route to Tier 2: pull the actual HOSE close for the dated session; a newspaper "hit 1,300" may be an intraday high, not the close — pin the exact value and date.
4. **Annotate** → *"Vietnam caps total foreign ownership of a domestic bank at 30% (Decree 01/2014/NĐ-CP). The VN-Index closed at 1,287 on 2026-05-23 (HOSE)."* Both now sourced, dated, and internally consistent.

## Phase-exit checklist

- [ ] Extractor run; CLAIM LEDGER turned into a worked table.
- [ ] Every POSSIBLE INTERNAL CONTRADICTION resolved or dismissed with a reason.
- [ ] NUMBER INDEX scanned — repeated figures are consistent; no unit drift.
- [ ] Every load-bearing figure cross-checked against its native tier, headline figures double-sourced.
- [ ] Every relative date ("last month", "recently") replaced with an absolute one.
- [ ] Live numbers carry an as-of date; deep-dives carry a `## Sources & further reading` section.
- [ ] No unverifiable figure left as false precision — each is sourced, ranged, dated, or cut.
- [ ] Re-ran the extractor after edits to confirm no new contradiction was introduced.
