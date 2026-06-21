---
title: "Building an SRE Culture and Team: Reliability Is a People Problem"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Every SLO and runbook in the world rots without the incentives, team topology, and dev-to-SRE contract to sustain it — here is how to staff reliability so it actually lasts."
tags:
  [
    "site-reliability-engineering",
    "sre",
    "sre-team-topology",
    "error-budget-policy",
    "dev-ops-contract",
    "production-readiness-review",
    "toil",
    "blameless-culture",
    "on-call",
  ]
category: "software-development"
subcategory: "Site Reliability Engineering"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/building-an-sre-culture-and-team-1.png"
---

I have watched two SRE teams with nearly identical headcount, tooling, and on-call rotations end up in completely different places. One team paid down its toil, kept availability climbing, and had engineers who'd been there four years. The other team — same Prometheus, same Grafana dashboards, same runbooks on the wiki — became a ticket queue, lost half its people in eighteen months, and shipped the *exact same outage* three quarters running. The tooling was not the difference. The difference was culture, incentives, and org structure: who carried the pager, what bar a service had to clear before SRE would support it, and whether leadership honored the freeze when the error budget blew.

This is the post where I argue, bluntly, that reliability is at least as much a people problem as a technical one. You can read the rest of this series — [the error budget as the currency of reliability](/blog/software-development/site-reliability-engineering/the-error-budget-the-currency-of-reliability), [toil as the silent tax on your team](/blog/software-development/site-reliability-engineering/toil-the-silent-tax-on-your-team), [designing a humane on-call](/blog/software-development/site-reliability-engineering/designing-a-humane-on-call), [the blameless postmortem](/blog/software-development/site-reliability-engineering/the-blameless-postmortem) — and execute every practice flawlessly, and still fail. The practices rot if the incentives are wrong. If devs throw unreliable code over a wall and never feel the 3am page, no SLO will save you. If the SRE team is a renamed sysadmin pool that catches everything, it will drown in toil and have no time to engineer the reliability it was hired for. If leadership treats the error budget as advisory the first time it's inconvenient, the budget is theater.

The figure below is the whole post in one image: the same headcount, the same tools, two outcomes. On the left, the renamed-sysadmins team — catching everything, 70% toil, burning out. On the right, the engineering team — protected by a bar to accept work, a toil cap, and automation that buys reliability back. By the end of this post you will be able to choose an SRE team topology that fits your org's size and maturity, write a dev-to-SRE contract that makes the engagement explicit, draft an error-budget policy that survives contact with a VP who wants to ship, recognize the ops-dumping-ground death spiral before it kills your team, and build the blameless culture that everything else rests on.

![Two SRE teams with identical headcount and tooling diverge into a drowning ticket queue versus a sustainable engineering team](/imgs/blogs/building-an-sre-culture-and-team-1.png)

Everything here ties back to the spine of the series: you define reliability with an SLI and SLO, measure it with observability, spend the error budget, reduce toil, respond to incidents, learn from them, and engineer the fix. That loop only runs if a real team, with the right incentives, is turning the crank. This post is about building that team. If you want the philosophy of why reliability is a feature you engineer rather than a wish, start at [the SRE mindset](/blog/software-development/site-reliability-engineering/reliability-is-a-feature-the-sre-mindset).

## 1. The thesis: incentives, topology, and the dev-to-ops relationship

Let me state the argument plainly so we can spend the rest of the post defending it.

**Reliability lives or dies on three things: incentives, team topology, and the relationship between the people who build software and the people who run it.** Get those three right and ordinary engineers produce reliable systems almost by default. Get them wrong and your best practitioners will quit, your best practices will rot, and you will relitigate the same outage forever.

Why those three and not, say, "better monitoring"? Because monitoring, runbooks, SLOs, and chaos drills are *outputs* of a healthy team. They are not causes of one. A team with the wrong incentives will let its dashboards go stale, will skip the postmortem follow-ups, will silence the noisy alert instead of fixing it. A team with the right incentives will keep all of that current because keeping it current is in their interest. So if you want durable reliability, you do not start by buying tools. You start by arranging the incentives so that the people closest to a problem are the people who feel its pain and have the authority to fix it.

Here is the incentive failure I see most often, stated as a single sentence: **the people who decide how reliable the software is are not the people who suffer when it breaks.** A developer ships a feature with a sloppy retry loop. At 3am it triggers a retry storm that takes down a downstream service. An SRE — someone who has never seen the code — gets paged, spends ninety minutes diagnosing it, mitigates by scaling up the downstream, and goes back to bed. The developer arrives at 9am, sees a green dashboard, and ships the next feature. Nothing in that loop teaches the developer to write a better retry. The pain landed on the wrong person. Until you fix that, you are subsidizing unreliability.

The fix is not "make developers feel pain for its own sake." It is *alignment*: arrange things so the cost of unreliability is felt by someone with the power and context to reduce it. Sometimes that means the developer carries the pager. Sometimes it means an SRE carries it but has the right to hand it back when a service is chronically broken. Either way, the loop has to close. We will spend most of this post on the mechanisms that close it — team topologies, the dev-to-SRE contract, the error-budget policy — but keep the underlying idea in view: **every mechanism in this post exists to route the pain of unreliability to a person who can do something about it.**

It's worth dwelling on *why* incentive misalignment is so corrosive, because it's not obvious until you've watched it happen. Engineers are not lazy or careless; they are *responsive to the signals they receive.* If the only signal a developer gets is feature velocity — shipped features are praised, the pager pain lands on someone else — then a rational, conscientious developer will optimize for velocity, and reliability will quietly degrade. Nobody decided to ship unreliable software. The org *structured* the incentives so that unreliable software was the locally rational output, and then acted surprised. This is why I distrust "just be more careful" as a reliability strategy. Carefulness is a personal virtue that doesn't survive contact with a misaligned incentive at scale. What survives is structure: a budget that makes the trade-off arithmetic, a pager that routes the pain, a policy that holds the line. You don't fix incentives with exhortation; you fix them with mechanism.

There's a useful way to test whether your incentives are aligned: ask, for any given reliability problem, *who feels it and who can fix it, and are they the same people?* When the answer is yes — the developer who wrote the flaky retry is the one woken up by it — the system self-corrects, because the person with the pain has the power. When the answer is no — the SRE feels it, the developer can fix it — the system rots, because the feedback never reaches the actuator. Every topology and contract decision in this post is really an answer to that one question, applied to a particular service. If you internalize nothing else, internalize the question: *feels-it equals fixes-it, or not?*

### Why this is the hardest part of SRE

The SLI-SLO-error-budget machinery is, frankly, the easy part. It is arithmetic and YAML. The hard part is organizational: getting leadership to honor a freeze when the budget's blown and the revenue feature is late; getting developers to accept a pager; protecting an SRE team's engineering time when there is an endless queue of "can you just look at this." These are political and cultural problems, and they do not yield to a Prometheus rule. They yield to pre-agreed policy, to executive sponsorship, and to a culture where reliability is funded as a priority rather than performed as a heroic side-effort. That is the territory of this post.

## 2. SRE team topologies: the spectrum from embedded to "you build it you run it"

There is no single correct way to organize reliability work. There is a *spectrum*, and where you sit on it should be a deliberate choice driven by your org's size, maturity, and how many services share the same reliability concerns. Let me lay out the four canonical points on that spectrum, then the trade-offs.

**Embedded SRE.** An SRE sits *inside* a single product team — same standup, same backlog, same Slack channel. They have deep context: they know the service's quirks, they were in the room when the architecture was chosen, they can spot a bad change in review. The downside is brutal arithmetic: one SRE per product team does not scale. If you have forty product teams you need forty SREs, and you almost never have them. There's also a subtler risk — the embedded SRE *goes native*. They become so identified with shipping the product team's features that they stop pushing back on reliability, and you've just paid a senior salary for another feature developer.

**Central or platform SRE.** A single central team builds reliability tooling and standards — the SLO framework, the alerting templates, the deploy pipeline, the incident process — and consults with product teams. This *scales*: one platform team can serve dozens of product teams because the leverage is in the tooling, not in human attention. The cost is the mirror image of embedded SRE's: the central team is *disconnected*. They don't live in any one service, so they make decisions with shallow, by-service context, and product teams experience them as a remote authority issuing standards from on high. The way you fight the disconnection is rotations and office hours, which we'll get to.

**Consulting or coach model.** The SRE provides *advice, not a pager*. They run production-readiness reviews, coach teams through their first incident, help design SLOs, and then step back. The team owns its own reliability; the SRE is a force multiplier who teaches rather than does. This scales beautifully — a handful of coaches can lift an entire engineering org. The risk is that advice without teeth is easy to ignore. A coach who can't gate a launch or hand back a pager has no leverage when a team decides reliability is optional.

**You build it you run it.** The far end of the spectrum: there is no separate SRE team for the service. The developers who build it carry its pager, run its incidents, and own its reliability end to end. This is the Amazon model, the modern DevOps end of the spectrum, and its great virtue is incentive alignment — the pain of a 3am page lands directly on the people who can change the code. The cost is that it demands mature developers who have the systems skills to run production, and it does not give you a pool of reliability *expertise* to draw on. A junior-heavy org doing pure you-build-you-run will reinvent every reliability wheel badly.

![A matrix comparing embedded, central, consulting, and you-build-you-run SRE models across whether they scale, their context, and their main risk](/imgs/blogs/building-an-sre-culture-and-team-2.png)

### When each fits

The honest answer is that **most real orgs are a hybrid**, and the right hybrid depends on three variables:

- **Org size.** A 20-engineer startup cannot afford a central SRE team and should default to you-build-you-run with a single reliability-minded senior engineer coaching. A 2,000-engineer company needs a central platform team or the tooling fragments into chaos.
- **Maturity.** Teams new to operating production need more support — embedded or heavy coaching — because you-build-you-run with engineers who've never carried a pager produces panic and bad mitigations. Mature teams can be handed the pager safely.
- **Criticality.** A service whose outage costs the business heavily justifies an embedded SRE or a high SRE-support bar. An internal batch job that can be down for a day does not — push it to you-build-you-run and spend your scarce SRE attention where the blast radius is large.

A common and healthy pattern: a **central platform team** that owns the shared reliability tooling and standards, which **embeds** an SRE temporarily into the two or three most critical, highest-traffic product teams, while **coaching** everyone else and pushing the long tail of low-criticality services to pure **you-build-you-run**. The platform team's leverage is the tooling; the embeds are a deliberate, time-boxed investment in the crown jewels; the coaching lifts the middle; and you-build-you-run handles the tail so SRE attention isn't wasted on services nobody would page for.

| Model | Org fit | Scales? | Context depth | Failure mode |
| --- | --- | --- | --- | --- |
| Embedded SRE | Small number of critical services | No — 1 per team | Deep | Goes native, no leverage |
| Central / platform SRE | Large org, many teams | Yes | Shallow, by-service | Disconnected ivory tower |
| Consulting / coach | Spreading reliability skills | Yes | Medium, project-based | No teeth, ignored |
| You build it you run it | Mature devs, low-to-mid criticality | Yes | Total | Reinvents wheels badly |

The decisive recommendation: **choose your topology per service, not per company.** Stop arguing about whether your company "is a DevOps shop" or "has SRE." Both. Map each service to a point on the spectrum based on its criticality and the team's maturity, and revisit the mapping when those change.

#### Worked example: mapping a 40-service estate

To make "per service, not per company" concrete, here is how I'd map a mid-sized estate of 40 services across the spectrum, and the headcount math that follows. Say the org has 12 product teams and a budget for 8 SREs.

- **3 crown-jewel services** (checkout, auth, the primary API) — high traffic, high criticality, an outage costs the business materially. These get **embedded SRE**: one SRE each, three SREs total. The leverage justifies the non-scaling cost.
- **The shared platform** (deploy pipeline, observability stack, SLO tooling) — built and run by a **central platform SRE** team of three. Their leverage is the tooling every other team uses, so three people serve all 40 services indirectly.
- **The middle tier of ~15 services** — moderately important, owned by mature-enough teams. These get **coaching**: the remaining two SREs run PRRs, office hours, and game days for them, and the teams own their own pagers. Two coaches lifting fifteen teams is real leverage.
- **The long tail of ~22 services** — internal tools, batch jobs, low-criticality glue. Pure **you build it you run it.** SRE provides the tooling (via the platform team) but carries no pager. Spending scarce SRE attention here would be poor leverage; these services can tolerate a slower mitigation.

Notice the arithmetic: 8 SREs cover a 40-service estate not by operating all 40 — which would need far more than 8 — but by *concentrating* attention where blast radius is high and *delegating* via tooling and coaching everywhere else. An org that tries to embed an SRE in every team needs 12 SREs and still hasn't built the shared platform. The hybrid is not a compromise; it's the only arrangement where the math closes. (Headcount and counts here are illustrative of the *shape* of the decision, not a prescription — your ratios depend on traffic, criticality, and team maturity.)

## 3. The "you build it you run it" spectrum and why the incentive matters

I want to spend a full section on the you-build-you-run versus dedicated-SRE-buffer question, because it is where the incentive argument bites hardest and where I see the most expensive mistakes.

Picture the same service two ways. In the first, a dedicated SRE buffer absorbs the pager. The developers ship; the SREs operate. The benefit is real: SREs bring expertise and focus, they're not context-switching between feature work and incidents, and they get genuinely good at running production. But there's a poison in the design. The developers no longer feel the consequences of shipping unreliable code. The 3am page lands on the SRE, not on them. Over time — and it does not take long — the developers optimize for the only signal they feel, which is feature velocity, and the reliability of what they ship drifts downward. They are, quite literally, throwing code over the wall, and the SRE buffer is the wall.

In the second arrangement, the developers carry their own pager. Now the incentive is aligned. A developer who ships a service that pages every night will be the one woken up every night, and that developer will, within about a week, find the energy to fix the thing that's paging. The pain of the page is the most honest reliability signal there is, and routing it to the person who wrote the code is the single most powerful incentive lever in this entire post.

![A before and after contrast showing a dedicated SRE buffer breaking the feedback loop versus full developer ownership routing the page back to the code authors](/imgs/blogs/building-an-sre-culture-and-team-3.png)

### Where to sit on the spectrum

So should everyone do pure you-build-you-run? No — and here is the nuance that the loudest DevOps advocates skip. Pure you-build-you-run has two costs. First, it scatters reliability expertise; nobody gets deeply good at running production because everyone's doing it part-time alongside feature work. Second, it can produce *worse* mitigations during incidents, because a developer who carries the pager twice a quarter has not built the muscle memory that an SRE who runs incidents weekly has.

The resolution is not to pick an end of the spectrum. It is to keep the *incentive* of you-build-you-run while adding the *expertise* of a dedicated function — and that is exactly what the dev-to-SRE contract in the next section does. SRE supports the service (expertise) *but only if it meets a bar* and *can hand the pager back* if the service is chronically unreliable (incentive). The hand-back lever is what stops a support relationship from degrading into a buffer. As long as SRE can return the pager, developers cannot throw code over the wall, because the wall can be removed.

The trade-off, laid out side by side, makes the design choice clear:

| Dimension | Dedicated SRE buffer | Full dev ownership | Supported with hand-back |
| --- | --- | --- | --- |
| Who feels the 3am page | SRE only | Developers | Developers if chronic, else SRE |
| Reliability expertise | Concentrated, deep | Scattered, shallow | Concentrated, shared via coaching |
| Incentive to write reliable code | Weak — pain is elsewhere | Strong — pain is direct | Strong — pager can return |
| Incident mitigation quality | High — SRE has the reps | Variable — devs out of practice | High while supported |
| Risk | Code thrown over the wall | Reinvented wheels, panic | Needs the hand-back enforced |

The right-most column is the synthesis this post argues for: a supported relationship with a real hand-back lever captures the expertise of a buffer and the incentive alignment of dev ownership at the same time, which is why the contract in the next section is built the way it is.

#### Worked example: the page count that fixed itself

A payments team I worked with ran a dedicated SRE buffer for a notification service. The service paged the SRE on-call roughly **18 times a week** — mostly transient timeouts that resolved on retry. The SREs had asked the dev team to fix the underlying timeout handling for three sprints; it never made the backlog, because the devs felt no pain. Velocity was their only signal.

We moved the notification service to you-build-you-run for one quarter as an experiment — the developers took the pager, with an SRE coaching them. In the *first week* a developer was paged four times and, by the end of that week, had shipped a fix: a sane timeout plus retry-with-backoff-and-jitter (the mechanics are in [timeouts, retries, and backoff done right](/blog/software-development/site-reliability-engineering/timeouts-retries-and-backoff-done-right)). Page count the following week: **3**. Within the quarter it settled at roughly **2 pages a week**, a drop from 18 to 2, an 89% reduction. Nothing changed except who felt the page. That is the incentive doing the work. (These figures are from one real engagement; treat the exact numbers as illustrative of the pattern, not a benchmark.)

## 4. The dev-to-SRE contract: PRR, error budget, and the hand-back

If your org has any kind of dedicated SRE function — embedded, central, or consulting — then the relationship between SRE and the dev teams it supports must be an *explicit contract*, not an implicit "SRE handles ops." The contract has three load-bearing parts: a bar to enter (the production-readiness review), a governor for the ongoing relationship (the error budget), and an exit clause (the right to hand the pager back).

![A graph of the dev to SRE contract showing a production readiness review gating support, the error budget governing the relationship, and a hand-back path when the budget is chronically blown](/imgs/blogs/building-an-sre-culture-and-team-4.png)

**The bar to enter: the production-readiness review (PRR).** SRE agrees to support a service *only if it meets a defined bar* of operability. The PRR is a checklist a service must pass before SRE will take its pager: it has meaningful SLOs, symptom-based alerts that page on user pain, runbooks for the top failure modes, dashboards that tell the truth, a tested rollback, capacity headroom, and a defined on-call. A service that hasn't done this work is not ready to be operated by anyone, and the PRR is how SRE refuses to inherit a mess. (The full PRR mechanics get their own post in this series — I'll link the planned `production-readiness-reviews` sibling — but the cultural point is the gate itself: support is *earned*, not assumed.)

**The governor: the error budget.** Once SRE supports a service, the error budget governs the day-to-day relationship without anyone having a meeting about it. While the service is inside its budget, developers ship freely — SRE does not get a veto over releases, because the budget is the permission. When the budget runs low, the relationship tightens: more caution on releases, reliability work prioritized over features. This is the whole reason the error budget is the *currency* of the series — it turns "is it reliable enough to ship?" into arithmetic both sides already agreed to. See [the error budget as the currency of reliability](/blog/software-development/site-reliability-engineering/the-error-budget-the-currency-of-reliability) for the budget math itself.

**The exit clause: the right to hand back.** This is the part most contracts omit and the part that makes the whole thing work. If a service is *chronically* unreliable — repeatedly blowing its budget, generating constant toil, ignoring the reliability work the budget demands — SRE has the contractual right to *hand the pager back* to the dev team. "Error budget exhausted, and stays exhausted, means SRE stops supporting and the dev team takes the pager until the service is healthy again." Without this clause, a support relationship is a one-way ratchet: SRE accumulates more and more chronically-broken services and can never shed any, which is precisely the death spiral we'll cover in section 6. The hand-back is the pressure-release valve.

### A support-agreement template

Here is a contract template I've used, written so a real team can adapt it. It's deliberately short — a support agreement people won't read is no agreement at all.

```yaml
# service-support-agreement.yaml
service: notification-api
owning_team: messaging
supporting_team: platform-sre
effective: 2026-Q3

# 1. The bar to enter (must all be true before SRE takes the pager)
entry_criteria:
  prr_passed: true              # production-readiness review signed off
  slos_defined: true            # at least availability + latency SLO
  alerts: symptom_based         # page on user pain, not on causes
  runbooks: top_5_failure_modes # linked, tested at last game day
  rollback: tested              # one-command, verified in staging
  oncall: defined               # named rotation, escalation path

# 2. What each side owns while supported
responsibilities:
  dev_team:
    - own service code and feature roadmap
    - keep runbooks current within 30 days of any change
    - join the incident bridge for Sev1/Sev2 on their service
    - complete postmortem action items by agreed dates
  sre_team:
    - carry primary pager for the service
    - maintain shared observability + deploy tooling
    - run quarterly game days and PRR refresh
    - advise on reliability design (consult, not gate)

# 3. The governor
error_budget:
  policy_ref: error-budget-policy.yaml   # see section 5
  in_budget: dev ships freely, no SRE veto
  low_budget: release caution + reliability work prioritized

# 4. The exit clause (the hand-back lever)
handback:
  trigger: budget exhausted 2 consecutive windows
  process:
    - SRE raises hand-back at the service review
    - 1 window grace to demonstrate recovery
    - if not recovered, pager returns to dev team
    - SRE re-engages after a passing PRR refresh
```

Notice what this does. It makes the *implicit* explicit. Most dev-SRE relationships fail not because anyone is acting in bad faith but because the expectations were never written down, so each side assumed a different deal. The dev team assumed SRE owned all operations forever; the SRE team assumed the dev team would fix the recurring pages. Writing it down — including the unglamorous hand-back clause — is how you avoid the slow-motion resentment that kills these relationships.

One more thing the contract should make explicit: *what SRE will not do.* A surprising amount of dumping-ground intake arrives as "can you just" — can you just run this query, can you just restart this nightly, can you just look at this customer's account. Each is small; together they are the toil that buries the team. A good support agreement names the categories of work SRE owns (the pager, the shared tooling, the reliability engineering) and, by omission and sometimes explicitly, the categories it does not (ad-hoc customer queries, manual deploys the dev team can self-serve, feature work). When a "can you just" arrives that falls outside the agreement, the SRE doesn't have to relitigate the relationship or feel like they're being difficult — they point at the contract. The contract is as much a *shield for the SRE* as it is a definition of service. Teams that skip writing down what SRE won't do find that the boundary erodes one reasonable-sounding request at a time, which is exactly how the spiral starts.

The contract also needs an owner and a review cadence, which is why the template carries an `effective` quarter and the support agreement is revisited at the service review. A contract written once and never revisited drifts out of date as the service changes, the team's maturity grows, and the criticality shifts. The quarterly service review is where you ask: is this still the right topology for this service? Has it earned a move from coaching to embedded, or from SRE-supported to you-build-you-run? Should the SLO change? The contract is a living agreement, not a founding document you sign once and frame.

### Stress-testing the contract

What if the dev team refuses to do the PRR work and demands support anyway? Then SRE declines, and the service runs you-build-you-run by default — which is fine, because an unready service shouldn't be operated by a team that lacks its context. What if a service blows its budget once due to a genuine black-swan event (a cloud-region outage)? The hand-back triggers on *consecutive* windows and a grace period precisely so a single bad break doesn't punish a healthy team — the budget accounting for whose fault an event is gets discussed in the error-budget post. What if leadership overrides the hand-back to keep SRE on a politically important service? Then leadership has just chosen to fund the toil, and that should be an explicit, budgeted decision, not a quiet imposition on the SRE team — which brings us to politics.

## 5. Error-budget politics: making the freeze stick

The error-budget *arithmetic* is easy. The error-budget *politics* is where SRE programs actually die. The hardest moment in this entire discipline is the one where the budget says "freeze" and a VP says "ship anyway." If you lose that argument, your error budget is decoration. If you win it, you have a functioning reliability culture. Everything in this section is about how to win it — and the answer is that you win it *before* the moment arrives, not during.

![A timeline of the quarter the error budget blew, showing a pre-signed policy making the feature freeze stick against a VP who wanted to ship anyway](/imgs/blogs/building-an-sre-culture-and-team-6.png)

**Why you can't win the argument in the moment.** When the budget blows and a launch is on the line, you are arguing against momentum: the feature is built, marketing has a date, the VP has promised it to *their* boss, and the SRE on the bridge is the only person saying no. In a live argument, seniority and volume win, not arithmetic. The SRE loses. So the trick is to never have the argument live. You have it once, in advance, when nobody is angry and no specific feature is at stake, and you write down what happens when the budget blows. Then, in the moment, you are not arguing — you are *invoking a policy that leadership already signed.*

### The error-budget policy

An error-budget policy is a short document, agreed and signed by both engineering and product/business leadership *in advance*, that specifies what happens at each budget threshold. The signature is the entire point. It converts a future fight into a past decision.

```yaml
# error-budget-policy.yaml — signed by VP Eng + VP Product, 2026-Q3
service: checkout-api
slo:
  availability: 99.9%        # 43.2 min/month of budget
  window: 30d rolling

# What happens at each budget level. Pre-agreed. Not negotiable in the moment.
thresholds:
  - budget_remaining: ">50%"
    policy: "Ship freely. No reliability gate on releases."
  - budget_remaining: "10%-50%"
    policy: "Ship, but every release runs full canary + extra review."
  - budget_remaining: "<10%"
    policy: >
      Feature freeze. Only reliability fixes, rollbacks, and
      security patches ship. Net-new feature launches are blocked.
  - budget_remaining: "exhausted"
    policy: >
      Hard freeze + reliability work is the team's top priority
      until budget recovers. SRE may invoke the hand-back clause.

# The escalation: who can override, and what it costs.
override:
  who: "VP Eng AND VP Product jointly, in writing"
  cost: "Override is logged, reviewed at the next ops review,
         and the unspent reliability work is added as debt."
  default: "Absent a signed override, the freeze holds."

# The sponsor: the exec who defends this when it's inconvenient.
exec_sponsor: "VP Engineering"
review_cadence: "quarterly, at the service review"
```

The override clause is important and is often misunderstood as a loophole. It is not a loophole; it is a *pressure gauge*. The policy does not say "leadership can never override the freeze" — that would be naive, because sometimes the business genuinely should accept reliability risk to ship something critical. It says the override must be made *jointly, in writing, by the people accountable for both velocity and reliability,* and it must be *logged and paid back.* This does two things. It raises the cost of an override enough that it's used only when it's genuinely worth it, and it makes the decision *visible* — when the same VP overrides the freeze three quarters running and the service is melting down, the logged overrides are the evidence that fixes the root cause: the policy, or the VP.

### The exec sponsor and tying reliability to the business

A policy with two signatures still needs a *defender* — an executive sponsor who will hold the line when a peer leans on them. The sponsor is usually the VP of Engineering, and their job in the politics is to absorb the pressure that would otherwise land on the SRE on the bridge. Without a sponsor, every freeze is a fresh fight at the IC level, and ICs lose fights with VPs.

The most durable way to give the sponsor ammunition is to **tie reliability to a business metric the VP already cares about.** "We were down 43 minutes" is abstract to a product VP. "We were down 43 minutes during peak checkout, which is roughly \$120,000 of abandoned carts, and the budget policy would have prevented the risky deploy that caused it" is not abstract. When reliability is denominated in the same currency as the features it's competing against, the freeze stops looking like engineers being precious and starts looking like risk management. This is also how you get reliability *funded* as a priority — staffed, on the roadmap, with allocated time — rather than performed as a heroic side-effort by whoever cares most. Heroics do not scale and the heroes burn out.

#### Worked example: two teams, one quarter, the budget blows

Two teams in the same org, same quarter, same kind of service. Team A signed an error-budget policy at the start of the quarter — both VPs signed it. Team B never got around to it; they had a budget and a dashboard but no signed policy.

Week 6, both teams hit trouble. A run of incidents spent **90%** of each team's 43.2-minute monthly budget. Week 7, both budgets hit **zero**. And in week 7, both teams had a revenue feature ready to launch, and in both teams a VP wanted to ship it on schedule.

Team A's SRE didn't argue. They pointed at the signed policy: budget exhausted means hard freeze, net-new launches blocked, override requires both VPs jointly in writing. The product VP could have requested the override — but doing so meant putting their name on a logged, reviewed decision to ship into a service that was already failing, *and* paying back the reliability debt. They declined. The launch slipped two weeks. The service recovered, the launch shipped into a stable system, and nobody got paged at 3am for it.

Team B's SRE *argued*. There was no policy to point at — only a dashboard and a strong opinion. The VP said the date was committed and the feature was important and surely one risky deploy wouldn't hurt. The SRE, outranked and out-argued, lost. The feature shipped into a service with no budget left. It triggered an incident two days later — a Sev1, a real outage, customer-visible — that cost far more in trust and in the ensuing scramble than the two-week slip would have. The postmortem's contributing factor was, in plain language, "shipped into an exhausted budget against the on-call's objection, no policy to enforce the freeze."

The difference between Team A and Team B was *one signed document, agreed before anyone was angry.* That is error-budget politics in one story. The budget is only a currency if leadership agrees, in advance, to honor it.

## 6. The ops-dumping-ground anti-pattern: the number-one way SRE fails

If I had to name the single most common way SRE programs fail, it is this: **the SRE team becomes a glorified ops team that catches everything developers throw over the wall.** The team drowns in toil, has no time to engineer, and burns out. This is the renamed-sysadmins failure, and it is so common that I'd estimate the majority of "we tried SRE and it didn't work" stories are really "we renamed our sysadmins and changed nothing else."

Let me describe the death spiral precisely, because recognizing it early is most of the cure.

![A graph showing the ops dumping ground death spiral feeding itself and three levers that break the loop and recover the team](/imgs/blogs/building-an-sre-culture-and-team-8.png)

It starts innocently. A new SRE team is stood up and, eager to prove its value, says yes to everything. A dev team has a flaky service? SRE will operate it. A manual deploy needs running every Friday? SRE will do it. A customer ticket needs a database query? Send it to SRE. Each "yes" is individually reasonable and individually small. But there is no bar to entry, so the intake is *unbounded*. The toil — the manual, repetitive, automatable, no-enduring-value work — climbs. (If you haven't internalized the precise definition of toil and why it's a tax, read [toil, the silent tax on your team](/blog/software-development/site-reliability-engineering/toil-the-silent-tax-on-your-team).)

Now the spiral feeds itself. As toil climbs past 50%, then 60%, then 70% of the team's time, there is no time left to *engineer* — no time to build the automation that would reduce the toil. So the toil never goes down; it only grows as more services pile on. The engineers who joined to build reliable systems are instead running manual deploys and answering tickets. They get bored, then frustrated, then they leave. Attrition means the remaining people carry even more toil, which deepens the spiral. The services SRE inherited are still unreliable — nobody had time to fix the root causes — so they keep generating the same incidents, quarter after quarter. The team is now a ticket queue with a fancier name, and reliability is no better than before SRE existed. Often it's worse, because the dev teams have stopped caring at all — that's SRE's job now.

### The warning signs checklist

Catch this early. Here are the signs your SRE team is becoming an ops dumping ground:

- **Toil is above 50% of the team's time** and trending up, not down. (Measure it — survey the team, or tag tickets. If you're not measuring toil you can't see the spiral.)
- **There is no bar to accept a service.** Anyone can hand SRE a pager and SRE always says yes.
- **SRE cannot say no, and cannot hand anything back.** The intake is one-way.
- **The roadmap is all reactive.** No proactive engineering projects survive; everything is firefighting and tickets.
- **The same incidents recur** quarter after quarter because nobody has time to fix root causes.
- **Developers have disengaged from reliability** — "that's SRE's job" is a sentence you hear.
- **On-call is dreaded**, the rotation is short-staffed, and people are quietly interviewing.
- **The team's name changed but nothing else did** — same people, same work, new title.

If you tick three or more of these, you are in the spiral. Here's how to climb out.

Before the cure, a word on *measuring* the spiral, because you can't manage what you don't measure and the toil percentage is the leading indicator. The honest way to measure toil is to have every SRE tag their time — even roughly, in a weekly self-report — into "toil" (manual, repetitive, automatable, reactive, no enduring value) versus "engineering" (building something that reduces future work). A small script over the team's tracked time tells you the trend, which is what matters more than the absolute number. Here's a minimal version you can adapt:

```python
# toil_trend.py — is the team in the spiral, or climbing out?
# Input: weekly self-reports of hours, tagged toil vs engineering.

def toil_fraction(week):
    toil = sum(h for kind, h in week if kind == "toil")
    total = sum(h for _, h in week)
    return toil / total if total else 0.0

def assess(weeks, cap=0.50):
    fractions = [toil_fraction(w) for w in weeks]
    latest = fractions[-1]
    trend = fractions[-1] - fractions[0]      # >0 means worsening
    over_cap = latest > cap
    spiraling = over_cap and trend > 0
    return {
        "latest_toil": round(latest, 2),
        "trend": round(trend, 2),
        "over_cap": over_cap,
        "spiraling": spiraling,
        "action": ("HAND BACK overflow until under cap"
                   if spiraling else "within healthy range"),
    }

# Example: six weeks, toil climbing past the 50% cap.
weeks = [
    [("toil", 22), ("engineering", 18)],   # 55%
    [("toil", 25), ("engineering", 15)],   # 62%
    [("toil", 27), ("engineering", 13)],   # 68%
    [("toil", 28), ("engineering", 12)],   # 70%
    [("toil", 29), ("engineering", 11)],   # 72%
    [("toil", 30), ("engineering", 10)],   # 75%
]
print(assess(weeks))
# -> {'latest_toil': 0.75, 'trend': 0.2, 'over_cap': True,
#     'spiraling': True, 'action': 'HAND BACK overflow until under cap'}
```

The point of the script is not the code; it's the *discipline* of turning toil from a vague feeling into a tracked number with a threshold and an automatic action. When `spiraling` flips true, the action is mechanical — hand overflow back — not a judgment call made in the moment by a tired team. The threshold does the deciding, exactly as the error budget does for releases. Now the cure.

### The four protections that break the spiral

The recovery is not subtle, but it requires organizational backbone:

1. **Install a bar to accept support — the PRR.** Stop catching everything. A service earns SRE support by passing the production-readiness review (section 4). Services that can't clear the bar run you-build-you-run. This stems the unbounded intake at the source.
2. **Cap toil and protect engineering time.** The canonical SRE rule is a **50% cap on toil** — at least half of every SRE's time is *protected* for engineering work that reduces future toil. This is not aspirational; it's enforced. If toil exceeds 50%, the overflow gets *handed back to the dev teams* until automation catches up. The cap is what guarantees there's always time to build the thing that shrinks the toil.

   Why exactly 50%, and why does the cap *break* the spiral rather than merely slow it? The arithmetic is worth doing. Toil grows roughly in proportion to the number of services and their inherent unreliability; engineering *reduces* toil by automating it away. If the fraction of time spent on engineering is $e$ and engineering retires toil at some rate proportional to $e$, then toil only trends down when $e$ stays above the level needed to out-pace the incoming toil. The moment engineering time hits zero — which is exactly what happens when toil reaches 100% — the toil can *never* decrease, because there's no one building the automation. So an uncapped team is on a one-way ratchet toward 100% toil, where it stays forever. The 50% cap forcibly reserves enough $e$ to guarantee toil can be paid down. It's not a comfort floor; it's the minimum engineering investment that keeps the system from locking up. A team allowed to drift to 80% toil has only 20% engineering, which is usually too little to out-run the intake, so it slides to 90%, then 100%, and locks. The cap is a circuit breaker for the toil spiral.
3. **Use the right to hand back.** Chronically unreliable services get their pager returned to the dev team (the hand-back clause from section 4). SRE is not a permanent home for broken services.
4. **Shift appropriate services to you-build-you-run.** Push the long tail of low-criticality and chronically-noisy services to developer ownership, where the incentive to fix them is strongest.

#### Worked example: the death spiral and the recovery

A real turnaround, numbers rounded but representative. An SRE team of six had drifted into the dumping ground. Measured toil was about **70%** of total team time. They supported **23 services**, most inherited without any bar. On-call generated roughly **35 pages a week**. Two of six engineers had left in the prior year; a third was interviewing. There were *zero* completed engineering projects in the last two quarters — everything was reactive.

The turnaround took three quarters and three levers:

- **Quarter 1 — install the bar and the cap.** They defined a PRR and applied it retroactively. Of the 23 services, **9 failed** the PRR badly enough that they were handed back to their dev teams to run (you-build-you-run) until they could pass. They instituted a hard 50% toil cap, enforced by handing overflow work back. Toil dropped from 70% to about **55%** as nine services left.
- **Quarter 2 — engineer down the toil.** With protected time finally available, the team automated the three worst toil sources: a manual weekly deploy (now a one-button Argo Rollout), a recurring database-failover runbook (now self-healing with a tested guardrail), and a class of ticket they'd been answering by hand (now a self-service tool). Toil dropped to about **40%**. Pages fell to roughly **15 a week** as the noisiest handed-back services were now the dev teams' problem and got fixed by the people who felt them.
- **Quarter 3 — sustainable.** Toil settled around **35%**, comfortably under the cap. Pages were down to about **6 a week**. The team shipped two proactive reliability projects. The engineer who'd been interviewing stayed. The metric that mattered most: the recurring quarter-over-quarter incidents finally stopped, because someone finally had time to fix the root causes.

Pages 35 to 6 a week. Toil 70% to 35%. Attrition halted. Same six people (well, four plus two backfills), same tooling. The difference was a bar, a cap, and the willingness to hand work back. (These figures come from a composite of real turnarounds and are illustrative of the pattern and its magnitude.)

The uncomfortable truth in this story is that the recovery *required handing nine services back to dev teams who didn't want them.* That is a political act, and it needs the same exec sponsorship as the error-budget freeze. An SRE team that cannot say no and cannot hand back is structurally guaranteed to end up in the spiral, no matter how good its engineers are. The protections are not optional niceties; they are the load-bearing structure.

## 7. Hiring and growing SREs: it is a software role

How you staff an SRE team encodes what you believe SRE *is*. If you hire pure operators — people whose only skill is responding to pages and running manual procedures — you will get the dumping ground, because operators without software skills cannot engineer away their toil; they can only absorb it. The single most important framing in SRE hiring is this: **SRE is a software engineering role applied to operations problems.** An SRE should be able to write the automation, build the tooling, and reason about systems — not just react to them.

### The blend of skills

A strong SRE sits at the intersection of two skill sets:

- **Software engineering.** Can write and review real code, build tools and automation, design APIs, reason about data structures and complexity. This is what lets an SRE *engineer down* toil rather than absorb it. The Google framing — that an SRE is a software engineer who also has the systems knowledge to run production — is the right north star.
- **Systems and operations.** Understands Linux, networking, distributed systems failure modes, databases, the kernel-to-cloud stack. Knows how things break in production, not just in theory. Can read a flame graph, follow a trace, reason about a saturated queue.

You rarely find both in equal measure in one hire, and that's fine. What you cannot accept is *neither the software skills nor a path to them*, because that hire can only ever be an operator, and operators feed the spiral. When interviewing, I weight the software-engineering signal heavily — a candidate who can code their way out of a toil problem is worth more than one who can hand-operate flawlessly, because the first one makes the second one's job obsolete.

### Growing SREs internally

The best source of SREs is often *inside your own engineering org.* A developer who has carried a pager, felt the pain, and wants to fix the systemic causes is an ideal SRE candidate — they already have the software skills and they've earned the operational scars. Rotations are the mechanism: rotate a developer into the platform SRE team for two quarters, or rotate an SRE into a product team. This fights the central team's disconnection problem (section 2) *and* grows reliability skills across the org *and* builds the relationships that make the dev-to-SRE contract work in practice rather than only on paper.

The rotation also solves a recruiting problem the market makes painful. Experienced SREs are scarce and expensive, and a job posting that demands "5+ years SRE experience plus strong software engineering plus deep distributed-systems knowledge" will sit open for months. Growing your own sidesteps the scarcity: you take a strong software engineer who's curious about production — there are many more of those — and you give them the operational reps via a rotation, with a senior SRE mentoring. After two quarters you have an SRE with both halves of the skill set *and* the institutional context that an external hire would take a year to acquire. The external hire still has a place — usually to bring in a missing specialty or to seed a new team with someone who's seen the patterns before — but the bulk of a healthy SRE org should be home-grown. A team that can only staff itself by external hiring is fragile; a team that grows its own is anti-fragile, because it converts its own incident pain into reliability talent.

There's a hiring anti-pattern worth naming explicitly: hiring for *firefighting heroism.* It is tempting to over-value the candidate who tells thrilling stories of saving the day during a Sev1 at 3am. Those stories are real and the skill is real, but if you optimize your hiring and your promotions for heroics, you get a team that is *good at outages* — and a team that's good at outages quietly prefers a world with outages to fight, because that's where the glory is. The reliability you actually want is *boring*: the systems that don't page, the toil that got automated away, the incident that didn't happen. Hire and promote for the boring win — the engineer who eliminated a class of pages so thoroughly that nobody remembers it was ever a problem — not for the dramatic save. What you reward is what you get more of, and you do not want more drama.

### The career path and the "software role" framing

If SRE is a dead-end ops job with worse hours than software engineering and no clear progression, your best people will leave and you'll be left with the spiral. So the career path has to be real and equivalent to the software-engineering ladder: an SRE should be able to reach staff and principal levels doing reliability work, evaluated on the systems they build and the reliability they enable, not on tickets closed or pages survived. The on-call has to be a *respected, sustainable* role — compensated, bounded (the humane-on-call practices in [designing a humane on-call](/blog/software-development/site-reliability-engineering/designing-a-humane-on-call)), and recognized as skilled work, not a hazing ritual. When SRE is framed and paid as a software engineering specialty, you attract software engineers, and software engineers build their way out of toil. When it's framed as ops, you attract operators, and operators drown in it. The framing is a hiring filter, and you choose which filter to run.

## 8. The culture that everything rests on

Topology, contracts, and policies are the skeleton. Culture is the soft tissue, and without it the skeleton doesn't move. Four cultural commitments underpin everything above, and the deepest one — the one the others rest on — is blamelessness.

![A stack showing blamelessness as the foundation under honest postmortems, toil treated as a bug, shared reliability ownership, and a respected on-call](/imgs/blogs/building-an-sre-culture-and-team-7.png)

**Blamelessness is the foundation — and it's not about being nice.** A blameless culture is one where, when something breaks, the question is "what about our systems and processes let this happen?" rather than "who screwed up?" This is not softness; it is *epistemics*. You only get the truth in a postmortem because nobody is punished for telling it. The moment an engineer believes that admitting "I ran the migration on prod by mistake" will get them fired, they will stop admitting it — and you will lose the single most valuable input to fixing the system: an honest account of what actually happened. Blame doesn't make people more careful; it makes them better at hiding. So blamelessness is the base layer of the culture stack because every other cultural good depends on the psychological safety it provides. The full mechanics of running a postmortem that surfaces truth are in [the blameless postmortem](/blog/software-development/site-reliability-engineering/the-blameless-postmortem); the cultural point here is that *the postmortem only works if the culture is already blameless.* You cannot bolt safety onto a blame culture with a template.

**Treat toil as a bug, not as the job.** In a healthy SRE culture, toil is not "what SREs do" — it is a defect in the system to be tracked, measured, and engineered away, exactly like a bug. When a manual procedure recurs, the cultural reflex should be "file a ticket to automate this," not "add it to the rotation's checklist." This reflex is what keeps the toil cap meaningful: the cap gives you the time, and the cultural framing makes you spend that time eliminating toil rather than getting better at enduring it.

**Reliability is everyone's job, not just SRE's.** The failure mode is "SRE owns reliability, so developers don't have to think about it." That is the dumping ground in cultural form. In a healthy org, reliability is a shared responsibility: developers write operable code, carry pagers or at least join bridges for their services, and treat their SLOs as a first-class part of the definition of done. SRE provides leverage — tooling, standards, expertise, coaching — but does not absolve everyone else of caring. The you-build-you-run incentive (section 3) is partly a cultural tool for exactly this: it makes reliability felt, and therefore owned, by the builders.

**Learning over blaming, and a respected on-call.** The orientation of the whole culture should be toward *learning* — incidents are expensive lessons you've already paid for, so extract every bit of value from them (this is the theme of [learning from incidents at scale](/blog/software-development/site-reliability-engineering/the-blameless-postmortem) and the broader incident-review practice). And the on-call has to be treated as a respected, sustainable, compensated role rather than a tax on the unlucky. When on-call is humane and respected, people will do it well; when it's a grind, your best people opt out and reliability degrades to whoever's left.

These four — blamelessness, toil-as-a-bug, shared ownership, respected on-call — are what make the structural mechanisms actually function. A signed error-budget policy in a blame culture will be used to assign fault for the budget burn, which poisons it. A toil cap in a culture that treats toil as "the job" will be ignored. The structure and the culture are not separable; you need both.

## 9. War story: how Google's error-budget model fixed the dev-versus-ops fight

The canonical real-world example, and the origin of much of this post, is Google's SRE model as documented in the Google SRE Book and SRE Workbook. I'll recount it accurately because it's the clearest illustration of incentives-by-design.

Google had the classic dev-versus-ops standoff: developers wanted to ship features fast; the ops/SRE side wanted stability and resisted change, because every change was risk they'd carry the pager for. Each side's incentive was locally rational and globally destructive — devs optimized for velocity, ops optimized for "nothing changes," and the two fought continuously over every release. The insight that broke the standoff was to make reliability a *budget* rather than a goal. Define an SLO (say 99.9% availability); the complement (0.1%, or about 43.2 minutes a month) is the error budget — the amount of unreliability you're *allowed* to spend. As long as the service is inside its budget, developers ship as fast as they want and SRE doesn't object, because the budget is the permission. When the budget's blown, releases slow and reliability work takes priority. Suddenly the two sides aren't fighting; they're reading the same number.

The Google model also documents the engagement structure this post is built on: SRE supports a service only if it meets a production-readiness bar; SRE *can hand the pager back* to the dev team if a service is chronically unreliable; and there's a cap on toil (the famous 50% rule) to keep SRE an engineering function rather than an ops queue. These aren't arbitrary policies — they're the structural defenses against the dumping-ground spiral, learned the hard way at scale. The error budget, the PRR, the hand-back, and the toil cap are a *coherent system* designed to keep incentives aligned and SRE engineering rather than drowning.

What's worth stressing — and what people miss when they copy the artifacts without the culture — is that none of this works without leadership buy-in. The error budget only governs releases because Google leadership agreed, in advance, that the budget would govern releases. The hand-back only works because the org agreed SRE could refuse chronically-broken services. Copy the YAML without the signed agreement and you get Team B from section 5. The model is as much an organizational agreement as a technical one. (The Google SRE Book and SRE Workbook chapters on engagement and embedding are the primary sources here; I'm summarizing their model, and any specific numbers like the 50% toil cap come from those texts.)

It's also worth being honest that Google's model was shaped by Google's scale and constraints, and it is not gospel for every org. Google had thousands of engineers, services with enormous blast radius, and the organizational maturity to honor a budget freeze — conditions that make a heavyweight SRE function pay off. A 50-person company copying the full apparatus — PRRs, embedded SREs, a formal error-budget policy with VP signatures — will spend more on the ceremony than they save in reliability. The *principles* travel: align incentives, make the budget arithmetic, protect engineering time, keep a hand-back lever. The *apparatus* should be sized to your scale. Take the incentive design; leave the bureaucracy you don't yet need. The companies that fail with SRE usually fail in one of two opposite ways — they copy the apparatus without the incentives (the rename), or they copy nothing and hope carefulness scales. Both are avoidable once you see that the model is a coherent incentive system you can scale up or down, not a checklist you adopt whole.

### A second, smaller story: the rename that changed nothing

A counter-example I've seen more than once: a company reads about SRE, likes the idea, and *renames its operations team to the SRE team.* New title, new org chart, same people, same work, same unbounded intake, no PRR, no toil cap, no hand-back, no error-budget policy. Six months later: "SRE didn't work for us." Of course it didn't — they adopted the *label* without any of the structural incentive changes that make SRE function. The lesson is the thesis of this entire post: SRE is not a job title or a tooling stack. It is a set of incentives and agreements. Change the incentives and you can call the team anything. Leave the incentives alone and renaming it SRE accomplishes precisely nothing.

## 10. The accept-this-pager decision, end to end

Let me pull the threads together into the one decision an SRE leader makes over and over: *should my team take responsibility for this service?* This is the operational form of everything above, and it's worth having a crisp mental model for it.

![A decision tree for whether SRE should carry a service's pager, splitting by readiness, by shared scale, and by whether it stays in budget](/imgs/blogs/building-an-sre-culture-and-team-5.png)

The decision splits cleanly along three gates, and the figure above is the tree:

**Gate 1 — readiness.** Did the service pass the PRR? If not, SRE does not take the pager; the service runs you-build-you-run until it earns support. No exceptions, because taking an unready service is how the dumping ground starts.

**Gate 2 — leverage.** Is this service shared by many teams, or is it one team's concern? A widely-shared, high-traffic, high-criticality service justifies central or embedded SRE attention because the leverage is high. A one-team, low-criticality service is better left to that team (you-build-you-run or coaching), because spending scarce SRE attention on it has poor leverage.

**Gate 3 — ongoing health.** Once supported, does the service stay inside its budget? If it chronically blows the budget, the hand-back clause fires and the pager returns to the dev team. Support is conditional on the service remaining operable.

Run every service through those three gates and you will never end up in the spiral, because the gates are precisely the protections: the readiness gate is the bar, the leverage gate keeps you focused on high-blast-radius services, and the health gate is the hand-back. The tree is the dumping-ground prevention, drawn as a decision.

### Stress-testing the whole system

Let me stress-test the way the series teaches. *What if two services blow their budgets the same quarter and both want overrides?* Then leadership is making two visible, logged, paid-back decisions, and the ops review will surface that the org is systematically under-investing in reliability — the politics scales because the policy scales. *What if the on-call is asleep when the budget-blown service breaks again?* The hand-back and the freeze are exactly what reduce that 3am page in the first place; a chronically-broken service shouldn't be on SRE's rotation generating those pages. *What if leadership refuses to sign the error-budget policy at all?* Then you don't have SRE; you have an ops team with nicer tools, and you should be honest about that internally rather than pretend the budget governs anything. *What if a critical service can't pass the PRR but the business needs it operated now?* Then leadership funds the gap explicitly — staffs the dev team to run it, or accepts the toil as a budgeted cost — rather than quietly dumping it on SRE. In every stress case, the resolution is the same: make the cost *visible* and put the decision with the people accountable for it. That's the meta-lesson of the whole post.

## 11. How to reach for this (and when not to)

Every structure has a cost, and a principal engineer's job is to know when *not* to build it. Here's the honest guidance.

**Reach for a dedicated SRE function when:** you have enough services and scale that reliability tooling and expertise have real leverage (roughly, when you have more than a handful of teams running production services); when your services are critical enough that outages cost the business materially; and when you have — or can get — the leadership buy-in to sign an error-budget policy and honor a hand-back. SRE without that buy-in is a costume.

**Do not build a separate SRE team when:** you're a small startup where every engineer can and should own production. A 15-person company that stands up a 2-person SRE team has just created a buffer that breaks the incentive loop and removed two builders. Do you-build-you-run with one reliability-minded senior engineer coaching, and revisit when you've got dozens of services.

**Do not adopt the embedded model at scale** — it doesn't scale, and you'll either go broke hiring or spread your SREs so thin they go native. Use it surgically for the two or three crown-jewel services and central/coaching for the rest.

**Do not adopt pure you-build-you-run with junior teams that have never operated production** — you'll get panic-driven mitigations and reinvented wheels. Pair it with heavy coaching until the teams have the muscle memory.

**Do not skip the error-budget policy because "we all agree reliability matters"** — you agree until the quarter the budget blows and the revenue feature is late, at which point unwritten agreements evaporate. The signature is the point.

**Do not rename your ops team to SRE and declare victory** — the most expensive non-change in this whole discipline. If you're not also changing the incentives (the bar, the cap, the hand-back, the budget policy), keep the old name; at least it's honest.

**Do not let SRE accept services with no bar** — the dumping ground is not a risk, it's the *default* outcome of unbounded intake. The bar is not bureaucracy; it's survival.

The recurring theme: the structures in this post are not free, and they're not always worth it. But when they're worth it, the *cheap* version — the rename, the buffer without a hand-back, the budget without a policy — is worse than nothing, because it costs real money and headcount while delivering the failure mode you were trying to avoid. If you're going to build it, build the load-bearing parts: the bar, the cap, the hand-back, the signed policy, and the blameless culture underneath. Skip any of those and you've built the spiral.

## 12. Key takeaways

- **Reliability is a people problem first.** Incentives, team topology, and the dev-to-ops relationship determine whether your SLOs and runbooks live or rot. The tooling is downstream of the team.
- **Route the pain to the people who can fix it.** Every mechanism here — you-build-you-run, the hand-back, the error budget — exists to make the cost of unreliability land on someone with the context and authority to reduce it.
- **Choose your topology per service, not per company.** Embedded for crown jewels, central/platform for leverage, coaching to spread skills, you-build-you-run for the tail. Most orgs are a deliberate hybrid; map each service by criticality and team maturity.
- **Make the dev-to-SRE relationship an explicit contract:** a bar to enter (the PRR), a governor (the error budget), and an exit clause (the right to hand the pager back). The hand-back is what stops a support relationship from decaying into a buffer.
- **Win the error-budget argument in advance.** A policy signed by engineering and product leadership *before* anyone is angry is what makes the freeze stick when the VP wants to ship. An unwritten agreement loses to seniority and volume every time.
- **The ops dumping ground is the default, not a risk.** Unbounded intake guarantees the toil spiral. The four protections — PRR bar, 50% toil cap, right to hand back, shift to you-build-you-run — are the load-bearing structure, not optional niceties.
- **SRE is a software engineering role.** Hire for the ability to engineer away toil, grow SREs internally via rotations, and make the career path and on-call respected — or you'll attract operators who can only absorb toil, not eliminate it.
- **Blamelessness is the foundation everything rests on.** You only get the truth in a postmortem because nobody's punished for telling it. A signed policy in a blame culture gets used to assign fault; structure and culture are inseparable.
- **The rename accomplishes nothing.** SRE is a set of incentives and agreements, not a job title or a tooling stack. Change the incentives and call the team anything; leave them alone and the title is a costume.

## 13. Further reading

- [Reliability is a feature: the SRE mindset](/blog/software-development/site-reliability-engineering/reliability-is-a-feature-the-sre-mindset) — the series intro and the philosophy this post operationalizes.
- [The error budget: the currency of reliability](/blog/software-development/site-reliability-engineering/the-error-budget-the-currency-of-reliability) — the budget arithmetic that governs the dev-to-SRE contract.
- [Toil: the silent tax on your team](/blog/software-development/site-reliability-engineering/toil-the-silent-tax-on-your-team) — what toil is, how to measure it, and why the 50% cap matters.
- [Designing a humane on-call](/blog/software-development/site-reliability-engineering/designing-a-humane-on-call) — making the on-call sustainable and respected so people do it well.
- [The blameless postmortem](/blog/software-development/site-reliability-engineering/the-blameless-postmortem) — why blamelessness surfaces truth, and how to run a review that learns.
- *The reliability maturity model* (planned sibling, `the-reliability-maturity-model`) — where your org sits on the journey from firefighting to engineered reliability, and what culture each stage needs.
- *Production-readiness reviews* (planned sibling, `production-readiness-reviews`) — the full PRR mechanics: the checklist, the sign-off, and how it gates SRE support.
- *Capstone: the SRE playbook* (planned, `capstone-the-sre-playbook`) — the whole field manual assembled into one operating model.
- The Google SRE Book and SRE Workbook (Google, free online) — the canonical source for the error-budget model, the engagement model, embedding, and the 50% toil cap. Chapters on "Embedding an SRE" and "How SRE Relates to DevOps" are the closest primary sources to this post.
- For the architecture-time side of reliability — designing systems that fail gracefully rather than running them — cross over to the system-design series on reliability, SLOs, and graceful degradation at `/blog/software-development/system-design/reliability-slos-error-budgets-and-graceful-degradation`.
