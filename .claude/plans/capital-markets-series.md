# Capital Markets — "Capital Markets: How Money Finds Its Best Use"

Series of **42 deep-dive posts** in a NEW folder **`content/blog/trading/capital-markets/`** (subcategory `Capital Markets`).
finance-writer voice, English, ≥ 8,800 words, ≥ 8 figures each, `.png` embeds + `optimize-blog-images`.
Audience: smart curious reader, **no finance background**. Global spine + a dedicated **Vietnam track**.
Kit: `.cache/finance-writer/_capital-markets/_kit.md` (reuse the `_macro` chartkit + a cited `data_capital.py`). Roadmap: this file.
Render: reuse `.cache/finance-writer/_macro/render-post.sh <slug>`. Commit + push **each wave** (explicit paths, never `git add -A`); scope to `trading/capital-markets/` slugs + their webp.

## Positioning (do NOT duplicate the sibling series — cross-link instead)
This series is **the capital-markets MACHINE itself** — the system that connects savers with companies and
governments — built from zero to practitioner depth, around one running thesis:

> A capital market is a **machine that turns savings into long-term investment**. It runs on two engines —
> a **primary market** that *creates* securities to raise capital, and a **secondary market** that *trades*
> them to provide liquidity — joined by a **plumbing layer** (clearing, settlement, custody) and run by
> **intermediaries**, all policed by **disclosure-based regulation**. The secret that makes it work:
> **secondary-market liquidity is what makes primary issuance possible** — nobody funds a 30-year project
> unless they can sell their claim on it tomorrow morning.

We own the **system / plumbing / lifecycle of a security** (issuance → trading → clearing → settlement →
regulation). We do NOT re-teach how to price or value the instruments. Link OUT:
- **fixed-income** owns bond pricing / duration / the curve → we own bond *issuance & deal mechanics*; link out.
- **equity-research** owns valuing a stock / DCF → we own how a stock gets *listed & traded*; link out.
- **options-volatility** owns derivatives trading / Greeks → link out.
- **banking** owns the commercial-bank balance sheet & lending → we own the *securities / investment-bank* side; link out.
- **hedge-funds** owns the buy-side firm internals → we own their *role as capital providers*; link out.
- **game-theory** owns the formal market-making / order-book / adverse-selection models → link out, keep the narrative.
- **macro-trading** owns policy / liquidity / the Fed → link out.
- **risk-management** owns risk frameworks → link out.
- **vietnam-stocks** owns picking VN stocks & sector rotation → we own VN market *structure & plumbing*; link out.
- Existing `finance` posts to BUILD ON (not duplicate): `inside-an-investment-bank-how-they-make-money`,
  `stock-exchanges-and-clearinghouses`.

## The savings→investment spine (woven through every track)
Every post connects its mechanics back to the spine: *a capital market exists to move savings into productive
investment, and it can only do that because the secondary market makes a long-term claim sellable today.*
Track A makes the spine explicit; the capstone ties the whole machine together.

---

## Track A — Foundations: What a Capital Market Is (6) — Wave 1
A1 `what-is-a-capital-market-how-money-finds-its-best-use` — Series intro + the savings→investment thesis; the two engines (primary/secondary); why a market beats a bank queue for allocating capital.
A2 `the-players-savers-issuers-and-the-middlemen-in-between` — The map: capital providers (households, funds, pensions) vs capital users (firms, governments) and the intermediaries (banks, exchanges, brokers) that connect them — and what each is paid for.
A3 `debt-vs-equity-the-two-ways-to-raise-capital` — The fundamental fork: sell a piece of ownership (equity) or borrow and promise to repay (debt); the claim each creates, the cost of each, and the capital-structure trade-off.
A4 `money-market-vs-capital-market-where-short-meets-long` — The maturity divide: overnight-to-1yr money market vs 1yr+ capital market; T-bills/CP/repo vs stocks/bonds; why the split matters for who funds what.
A5 `what-a-security-actually-is-claims-you-can-sell` — A security as a *standardised, transferable claim*; shares vs bonds vs units as legal claims; why standardisation + transferability is the whole invention.
A6 `the-life-of-a-security-from-idea-to-delisting` — The full lifecycle: a financing need → issuance → listing → secondary trading → corporate actions → maturity/buyback/delisting; the map for the whole series.

## Track B — The Primary Market: Raising Capital (6) — Wave 2
B1 `the-financing-ladder-from-bootstrap-to-public-markets` — How a company climbs from founders' cash → angels/VC → private rounds → the public market; what changes at each rung and why firms eventually tap public capital.
B2 `the-ipo-process-end-to-end-from-mandate-to-first-trade` — The full IPO: picking banks, due diligence, the S-1/prospectus, the roadshow, pricing night, allocation, and the opening cross on listing day.
B3 `underwriting-and-the-syndicate-who-takes-the-risk` — Firm-commitment vs best-efforts; the bookrunner and the syndicate; the underwriting spread/gross spread; the greenshoe (over-allotment) and stabilisation.
B4 `bookbuilding-and-price-discovery-how-the-ipo-price-is-set` — How demand becomes a price: the book, the range, indications of interest, allocation discretion, the deliberate "IPO pop", and why auctions (Google) are rare.
B5 `beyond-the-ipo-follow-ons-rights-issues-and-private-placements` — Raising more after listing: follow-on offerings, rights issues, accelerated bookbuilds, at-the-market programs, PIPEs, and the dilution each causes.
B6 `how-a-bond-is-issued-auctions-syndication-and-the-deal` — Selling debt at scale: government auctions (competitive vs non-competitive bids), corporate syndicated deals, the order book, new-issue concession, and the primary dealer system.

## Track C — The Secondary Market: Trading Infrastructure (6) — Wave 3
C1 `inside-an-exchange-the-matching-engine-and-the-order-book` — What an exchange really is: the central limit order book, price-time priority, the matching engine, the open/close auctions, and why a continuous market exists.
C2 `order-types-and-how-an-order-travels-to-the-market` — Market vs limit vs stop; the full order lifecycle from your app to the book; routing, the broker, and what "best execution" obligates.
C3 `market-makers-and-the-spread-who-provides-liquidity` — The liquidity supplier's business: the bid-ask spread as their pay, inventory risk, adverse selection, designated market makers vs electronic LPs/HFT.
C4 `lit-markets-dark-pools-and-the-fragmented-tape` — Where trading actually happens: lit exchanges, dark pools/ATSs, wholesalers/internalisers, payment for order flow, fragmentation, and the consolidated tape/NBBO.
C5 `how-a-price-is-made-discovery-arbitrage-and-efficiency` — Price formation from order flow; the role of arbitrage in keeping prices consistent; informational efficiency and its limits.
C6 `indices-etfs-and-the-bridge-back-to-the-primary-market` — How an index is built, index inclusion as a tradable event, and ETF creation/redemption — the mechanism that quietly links secondary trading back to primary issuance.

## Track D — Clearing, Settlement & Custody: The Plumbing (5) — Wave 4
D1 `what-happens-after-the-trade-the-post-trade-lifecycle` — The hidden second half: trade capture, affirmation/confirmation, clearing, settlement; why "the trade" is only the beginning and T+1/T+2 exists.
D2 `the-clearinghouse-how-a-ccp-removes-counterparty-risk` — Novation and multilateral netting; how a central counterparty steps between buyer and seller so neither can default on the other.
D3 `margin-and-the-default-waterfall-how-a-ccp-survives-a-blowup` — Initial vs variation margin, the guarantee/default fund, the loss waterfall, and skin-in-the-game — how the clearinghouse itself doesn't fail.
D4 `settlement-and-custody-who-actually-holds-your-shares` — CSDs, book-entry, delivery-versus-payment, omnibus vs segregated accounts, custodians and sub-custodians — the chain between "you own it" and the share register.
D5 `securities-lending-and-repo-the-financing-plumbing` — How short selling and financing actually work: the stock-loan market, the repo market, rehypothecation, and why this plumbing is the market's hidden funding layer.

## Track E — The Intermediaries: Sell-Side, Exchanges & Buy-Side (5) — Wave 5
E1 `inside-an-investment-bank-ecm-dcm-ma-and-trading` — The sell-side product map: equity & debt capital markets, M&A advisory, sales & trading, research; how each desk makes money and the fee pools they fight over.
E2 `the-broker-dealer-agency-principal-and-prime-brokerage` — Acting as agent vs trading as principal; retail brokers, institutional brokers, prime brokerage for hedge funds, and the conflicts baked into each role.
E3 `the-buy-side-who-actually-owns-the-market` — Asset managers, pension funds, insurers, mutual funds/ETFs, hedge funds, sovereign wealth funds — the capital providers, their mandates, and how their size shapes the market.
E4 `sell-side-research-the-analyst-the-rating-and-the-wall` — What research is for, the buy/hold/sell rating, the information barrier ("Chinese wall"), MiFID II unbundling, and the conflicts around IPO research.
E5 `the-supporting-cast-exchanges-data-indices-and-ratings` — The for-profit ecosystem: exchanges as listed businesses, market-data revenue, index providers, credit-rating agencies, and proxy advisors — who sells the picks and shovels.

## Track F — Securitization & Structured Capital (5) — Wave 6
F1 `securitization-from-first-principles-turning-loans-into-bonds` — The core trick: pool illiquid loans, sell the cash flows as tradable securities; the SPV/bankruptcy-remoteness, tranching, and why it lowers the cost of capital.
F2 `abs-and-mbs-the-mortgage-and-consumer-credit-machine` — Asset- and mortgage-backed securities: pass-throughs, the cash-flow waterfall, prepayment risk, agency vs private-label, and the scale of the market.
F3 `cdos-clos-and-the-tranching-of-tranches` — Re-securitization: CDOs and CLOs, the senior/mezz/equity stack, the correlation assumption, and how AAA gets manufactured from BBB.
F4 `covered-bonds-abcp-and-the-shadow-funding-chain` — The other structured-funding tools: covered bonds (on balance sheet), asset-backed commercial paper, conduits/SIVs, and the maturity-transformation chain outside the banks.
F5 `2008-when-the-securitization-machine-broke-case-study` — How the plumbing failed: the subprime chain, ratings on flawed models, the run on repo/ABCP, AIG and the CCP-less CDS market, and the capital-markets lessons.

## Track G — Regulation, Disclosure & Market Integrity (4) — Wave 7
G1 `why-markets-are-regulated-disclosure-and-the-securities-acts` — Disclosure-based regulation: the 1933 (issuance) and 1934 (trading) Acts, the birth of the SEC, "sunlight is the best disinfectant", and registration vs exemption.
G2 `disclosure-the-prospectus-filings-and-insider-trading` — Continuous disclosure: the prospectus, 10-K/10-Q/8-K, materiality, Reg FD, and why insider trading law exists to protect the disclosure regime.
G3 `market-integrity-manipulation-spoofing-and-circuit-breakers` — Keeping the game fair: manipulation, spoofing/layering, front-running, the surveillance layer, circuit breakers and limit-up/limit-down after the 2010 flash crash.
G4 `global-market-structure-reg-nms-mifid-and-cross-border` — How the rules differ across the world: US Reg NMS vs EU MiFID II, IOSCO standards, listing regimes, and what cross-border listing/dual-listing involves.

## Track H — Vietnam's Capital Market (4) + Capstone (1) — Wave 8
H1 `vietnams-capital-market-hose-hnx-upcom-and-the-ssc` — The architecture: the two exchanges + UPCoM, the holding company VNX, the State Securities Commission as regulator, and how Vietnam's young market is organised.
H2 `clearing-and-settlement-in-vietnam-vsdc-and-the-krx-system` — The plumbing locally: the Vietnam Securities Depository and Clearing Corp, the T+2 cycle, the pre-funding problem, the move to a CCP, and the KRX trading-system upgrade.
H3 `foreign-flows-and-ownership-limits-the-room-and-the-workarounds` — Foreign ownership limits ("the room"), how foreign investors access the market, NVDR-style workarounds, and why foreign flows move the VN-Index.
H4 `the-emerging-market-upgrade-what-vietnam-must-fix` — The FTSE Russell / MSCI upgrade story: the frontier→emerging criteria, the pre-funding and FOL blockers, the reforms underway, and what an upgrade would unlock.
CAP `the-capital-markets-machine-the-whole-system-end-to-end` — Capstone: walk one dollar of savings through the entire machine — issuance → trading → clearing → settlement → custody → back to new issuance — tying every track together.

---

## Cross-links the kit hands to agents (real, existing posts — confirm siblings as waves ship)
- Bond instruments (don't re-price): `/blog/trading/fixed-income/how-bonds-are-priced-present-value-of-future-cash-flows`,
  `/blog/trading/fixed-income/the-yield-curve-explained-the-most-important-chart-in-finance`,
  `/blog/trading/fixed-income/emerging-market-and-sovereign-debt-yield-with-country-risk`
- Equity valuation (don't re-value): `/blog/trading/equity-research/` (link a valuation/DCF post as a sibling check)
- Investment bank / exchange (BUILD ON): `/blog/trading/finance/inside-an-investment-bank-how-they-make-money`,
  `/blog/trading/finance/stock-exchanges-and-clearinghouses`
- Market-making / order-book model: `/blog/trading/quantitative-finance/order-book-simulator-quant-research`,
  `/blog/trading/quantitative-finance/market-making-simulator-quant-research`,
  `/blog/trading/game-theory/` (adverse-selection / Glosten-Milgrom / Kyle posts)
- Banking / securitization overlap: `/blog/trading/banking/securitization-how-banks-turn-loans-into-securities`
- Buy-side: `/blog/trading/hedge-funds/` (a fund-structure / prime-brokerage post)
- Macro / liquidity: `/blog/trading/macro-trading/` (a Fed / liquidity post)
- Vietnam: `/blog/trading/vietnam-stocks/foreign-flows-etfs-and-the-index-effect-vietnam`
- Crises / case: `/blog/trading/finance/ltcm-1998-when-genius-failed`, `/blog/trading/finance/asian-financial-crisis-1997`
Only link posts you are reasonably sure exist (siblings in this series, or the BUILD-ON pair above).

## Build state
- [ ] Wave 1 (Track A, 6) — NOT STARTED
- [ ] Wave 2 (Track B, 6)
- [ ] Wave 3 (Track C, 6)
- [ ] Wave 4 (Track D, 5)
- [ ] Wave 5 (Track E, 5)
- [ ] Wave 6 (Track F, 5)
- [ ] Wave 7 (Track G, 4)
- [ ] Wave 8 (Track H, 4 + capstone)
Total: 42 posts.
