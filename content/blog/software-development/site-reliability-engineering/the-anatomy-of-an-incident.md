---
title: "The Anatomy of an Incident: The Lifecycle That Keeps a Team Calm"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Walk an incident through its real lifecycle — detect, declare, triage, mitigate, diagnose, resolve — learn a severity scale that routes the response automatically, and build the timeline that feeds MTTR and the postmortem."
tags:
  [
    "site-reliability-engineering",
    "sre",
    "incident-response",
    "incident-management",
    "severity",
    "mttr",
    "on-call",
    "incident-lifecycle",
    "reliability",
  ]
category: "software-development"
subcategory: "Site Reliability Engineering"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/the-anatomy-of-an-incident-1.png"
---

It is 02:14. The pager goes off. Your checkout error rate is climbing and you do not know why yet. Here is the thing that separates the teams that recover in twenty minutes from the teams that flail for two hours: it is not who has the smartest engineer awake, and it is not who has the fanciest dashboard. It is whether everyone in the channel knows *which stage of the incident they are in*. A team that knows the lifecycle stays calm, because at every moment there is exactly one right next move and everyone can name it. A team that does not know the lifecycle improvises, and improvisation at 2am is how a fifteen-minute blip becomes a war-room legend.

An incident is not a formless emergency. It has an anatomy — a predictable sequence of stages, each with an owner, an entry condition, and an exit condition. You move through them in order: you **detect** something is wrong, you **declare** that this is an incident, you **triage** to decide how bad it is, you **mitigate** to stop the bleeding, you **diagnose** the actual cause, and you **resolve** when the system is genuinely healthy again. Then you hand off to the **learn** stage — the blameless postmortem — which we cover in its own post. The whole point of naming these stages is that under stress, human beings lose the ability to plan; they need a checklist they can fall back on. The lifecycle *is* that checklist.

![A flow diagram of the incident lifecycle showing detect, declare, triage, mitigate, diagnose, and resolve with declaring early as a cheap gate that beats limping on alone](/imgs/blogs/the-anatomy-of-an-incident-1.png)

This post is the operational heart of the response track. By the end you will be able to do five concrete things. You will recognize each stage of an incident and know who does what in it. You will declare incidents earlier, because you will believe — correctly — that declaring is cheap and limping is expensive. You will assign a severity from a written scale instead of arguing about it, and that severity will automatically route who gets paged, who gets notified, and how often you update people. You will reach for mitigation *before* diagnosis, because the user feels the bleeding, not the root cause. And you will produce a clean incident timeline that feeds your MTTR metric and seeds the postmortem. This sits squarely in the *respond* phase of the series spine — define reliability, measure it, budget it, respond to incidents, learn, and engineer the fix — and it leans on the SRE mindset laid out in [reliability is a feature](/blog/software-development/site-reliability-engineering/reliability-is-a-feature-the-sre-mindset). Let us walk the anatomy.

## 1. The incident lifecycle: six stages, one calm checklist

Start with the shape of the whole thing, because every later section is a zoom into one stage. An incident moves through six stages in order, and a seventh — learning — happens afterward. Here they are with their one-line jobs.

- **Detect.** Something is wrong, and a signal tells you so: an alert fires, a customer complains, a metric crosses a line. The clock for *mean time to detect* (MTTD) starts when the problem begins and stops when a human knows about it.
- **Declare.** A human says, out loud and in writing, "this is an incident." Declaring is a deliberate act. It spins up a dedicated channel, names an incident commander (IC), and starts a communication cadence. Declaring is cheap. Not declaring — limping along with three people quietly poking at production in DMs — is expensive.
- **Triage.** You answer one question fast: how bad is this? The answer is a *severity* level (Sev1 through Sev4), and that severity is not a vibe — it is a lookup against user impact and scope.
- **Mitigate.** You stop the bleeding. Mitigation is anything that restores service to users, whether or not you understand the cause: roll back the deploy, fail over to the healthy region, shed load, flip the feature flag off. Mitigation has priority over diagnosis. Always.
- **Diagnose.** *Now* you find the root cause. You can afford to be careful here precisely because users are no longer in pain — mitigation bought you the time.
- **Resolve.** You confirm the system is genuinely healthy against explicit all-clear criteria, record the final timeline, and close the incident. The clock for *mean time to resolve* (MTTR) stops here.
- **Learn.** You run a blameless postmortem, extract the contributing factors, and file the action items that prevent a repeat. This is the handoff to [the blameless postmortem](/blog/software-development/site-reliability-engineering/the-blameless-postmortem).

The single most important property of this list is that it is *ordered*, and the order encodes hard-won judgment. Triage comes before mitigation so you size your response correctly. Mitigation comes before diagnosis so users stop suffering while you investigate. Resolution comes with explicit exit criteria so nobody declares victory on a hunch. When a team internalizes this order, the 2am scramble becomes a calm sequence of "what stage are we in, what is the next move." When they have not, you get the classic failure mode: six engineers all diagnosing at once, nobody mitigating, no IC, and the error rate climbing the whole time.

> The lifecycle is a forcing function. Each stage has a clear owner and a clear next action, so under stress nobody has to invent the playbook — they just have to follow it.

The rest of this post walks each stage, defines the severity scale that triage produces, defines the four "mean time to" metrics that the timeline produces, and runs two worked examples — a textbook Sev2, and a severity mis-classification that taught a real lesson. For the *who keeps the room calm* dimension — running the bridge, the roles, the comms — see [incident command, staying calm under fire](/blog/software-development/site-reliability-engineering/incident-command-staying-calm-under-fire). For the *why mitigation beats diagnosis* dimension in depth, see [mitigate first, diagnose later](/blog/software-development/site-reliability-engineering/mitigate-first-diagnose-later). This post is the map that ties them together.

## 2. Detect: the clock starts when the problem starts, not when you notice

Detection is the first stage and it has a brutal property that people misunderstand: the detection clock does not start when your alert fires. It starts when the *problem* starts. If a bad deploy began corrupting checkout at 02:00 and your alert fired at 02:14, your MTTD for that incident is fourteen minutes — fourteen minutes of users hitting errors that you knew nothing about. Every one of those minutes added to the blast radius.

That is the principle of detection: **faster detection shrinks the blast radius.** Blast radius is the set of users and data affected by a failure, and it almost always grows monotonically with time. A region failover that you catch in two minutes affects a thin slice of in-flight requests; the same failover caught in forty minutes has timed out thousands of sessions, filled retry queues, and possibly cascaded into dependent services. The whole reason SRE teams invest so heavily in alerting and observability is that the cheapest minute of an incident is the one you never have to fix, and the second cheapest is the one you detect immediately.

Detection comes from three sources, and a mature team wires up all three.

| Detection source | What it catches | Typical MTTD | Weakness |
| --- | --- | --- | --- |
| Symptom alert on an SLI | User-facing pain (error rate, latency, availability) | Seconds to ~2 min | Only as good as your SLIs |
| Synthetic / probe check | Whole user journeys, even with zero live traffic | 30 s to a few min | Can miss real-traffic-only bugs |
| Human report (user, support, exec) | Anything your monitoring missed | Minutes to hours | Slow, embarrassing, lossy |

The order of that table is the order you want detection to happen. The best case is a symptom-based alert: you page on user pain — "checkout success rate dropped below 99%" — not on a possible cause like "CPU is high." A symptom alert fires once, for the thing users actually feel, and it tells you the blast radius directly. The art of writing alerts that page on symptoms and do not cry wolf is its own discipline; see [alerting that doesn't cry wolf](/blog/software-development/site-reliability-engineering/alerting-that-doesnt-cry-wolf) and the foundation of picking the right signal in [choosing SLIs that reflect user pain](/blog/software-development/site-reliability-engineering/choosing-slis-that-reflect-user-pain). The worst case — the one that should make you wince — is detecting an outage because a customer tweeted at you. If your MTTD source is "Twitter," your detection is broken.

Here is the kind of symptom alert that drives good detection. It pages when the rolling checkout success ratio drops, computed as good events over total events — the standard SLI shape.

```yaml
groups:
  - name: checkout-sli
    rules:
      # Recording rule: the SLI as a good-over-total ratio, 5m window
      - record: checkout:success_ratio:rate5m
        expr: |
          sum(rate(http_requests_total{job="checkout",code!~"5.."}[5m]))
          /
          sum(rate(http_requests_total{job="checkout"}[5m]))

      - alert: CheckoutSuccessRateLow
        expr: checkout:success_ratio:rate5m < 0.99
        for: 2m
        labels:
          severity: page
          team: payments
        annotations:
          summary: "Checkout success rate {{ $value | humanizePercentage }} (< 99%)"
          runbook: "https://runbooks.internal/checkout-success-low"
```

Two things make this a detection asset rather than a noise source. First, it is a *symptom*: it watches the success ratio users experience, so when it fires you already know roughly how bad it is. Second, the `for: 2m` clause means the condition must persist for two minutes before it pages, which suppresses single-scrape blips while costing you only two minutes of MTTD. That trade — a touch of latency for a lot less noise — is almost always worth it for a page-grade alert.

#### Worked example: what one fast minute of detection is worth

Suppose checkout processes 1,000 requests per minute and a bad deploy makes 30% of them fail. Every minute of undetected failure means roughly 300 failed checkouts. If your MTTD is 14 minutes, that is about 4,200 failed checkouts before anyone knew. Cut MTTD to 2 minutes with a good symptom alert and you are at about 600 failed checkouts before the page — a 7x reduction in blast radius *before you have even started responding*. Detection is the single highest-leverage minute in the whole incident, which is why an SRE who is serious about MTTR obsesses about MTTD first. At an average margin of \$8 per checkout, that 3,600-checkout difference is on the order of \$28,000 of avoided lost revenue, and that is before counting the reputational cost of users who never come back.

There is a deeper reason the detection minute is worth so much, and it is worth stating plainly because it changes where you spend your reliability budget. Most of the harm in an incident does not accumulate linearly — it accumulates *super-linearly*, because slow detection lets the failure interact with the rest of the system. In the simple checkout example the cost grew at a flat 300 failed checkouts per minute, which already justifies fast detection. But real systems rarely fail in isolation. The failing checkout service times out, its callers retry, the retries pile onto a database that is already struggling, the database slows down further, and now a checkout problem has become a database problem that touches search, recommendations, and the account page too. Every minute you do not detect is a minute the failure has to find new surface area to spread across. The flat-rate arithmetic is the *floor* on the cost of slow detection; the real cost has a long, ugly tail that the arithmetic does not capture.

### Detection: entry and exit criteria

It helps to be precise about when the detect stage *begins* and when it *ends*, because those two moments anchor MTTD and they are surprisingly easy to get wrong. The **entry condition** for the detect stage is the onset of the problem itself — the first request that failed, the first byte of corrupted data written, the first moment the SLI departed from healthy. This is almost never the moment a human looks at it; it is a fact about the system that you reconstruct *afterward* from logs and metrics. The **exit condition** for detect is the moment a human (or an automated responder) is aware that the problem exists and is real — not a flapping alert that auto-resolved, but a signal a person has registered. The gap between those two is MTTD, and the honest version of that number requires you to go back into the data after the fact and find the true onset, not to start the clock when the pager buzzed. Teams that measure MTTD from "alert fired" systematically flatter themselves, because they are excluding the very window — onset to alert — that they are trying to shrink.

A second subtlety in detection is the difference between a *signal* and an *incident*. Not every alert is an incident, and not every incident starts with an alert. An alert is a hypothesis that something might be wrong; the detect stage is complete only when that hypothesis has been confirmed as a real, ongoing problem worth responding to. This is why the cleanest detection pipelines separate "alert fired" from "incident detected" — the alert is the input, and a human (or, increasingly, an automated triage rule) turns it into a confirmed detection. A flapping alert that fires and clears in forty seconds with no user impact is a tuning problem, not a detected incident, and conflating the two is how MTTD gets polluted with noise that makes the trend line meaningless.

## 3. Declare: declaring is cheap, limping is expensive

This is the stage teams skip, and skipping it is the most common reason a small problem becomes a big one. *Declaring* an incident is the explicit act of saying "this is an incident" and triggering the machinery: a dedicated incident channel is created, an incident commander is named, and a communication cadence begins. The opposite — the trap — is what I call *limping*: a couple of engineers notice something is off and start quietly debugging in a thread or a DM, never declaring, never naming an owner, never telling anyone.

![A before and after comparison of an improvising team that diagnoses first and burns ninety minutes versus a team that declares early and recovers in thirty-seven minutes](/imgs/blogs/the-anatomy-of-an-incident-2.png)

The principle here is an asymmetry of cost, and once you see it you will never hesitate to declare again. **Declaring is cheap; limping is expensive.** Take the case where you declare and it turns out to be nothing: you spent five minutes, created a channel you will archive, briefly pulled in an IC who stands down, and you write a two-line "false alarm" note. Total cost: a few minutes and mild embarrassment. Now take the case where you *should* have declared and did not: the problem grows while three people poke at it without coordination, nobody owns the decision to roll back, leadership finds out from a customer, and there is no timeline because nobody was writing anything down. The expected cost of under-declaring dwarfs the expected cost of over-declaring, so the correct policy is a strong **bias to declare.** When in doubt, declare. You can always downgrade or stand down.

Declaring does three concrete things, and each one is a direct antidote to a way that incidents go wrong:

- **It creates a single channel.** All discussion, all timestamps, all decisions happen in one place. No more "wait, which thread is this in?" The channel becomes the source of truth and, later, the raw material for the timeline.
- **It names an incident commander.** The IC owns the response — not the fix, the *response*. The IC decides, delegates, and keeps the room calm. One person is clearly in charge, which kills the "everyone is responsible, so no one is" failure mode. The IC role is deep enough to deserve its own treatment; see [incident command, staying calm under fire](/blog/software-development/site-reliability-engineering/incident-command-staying-calm-under-fire).
- **It starts a communication cadence.** From the moment you declare, stakeholders get updates on a schedule — every 30 minutes for a Sev1, say. This is not bureaucracy; it is what stops the IC from being interrupted every two minutes by "any update?" pings, and it is what keeps the status page honest.

Declaring should be a single, low-friction command. Most teams wire it into chat. Here is the shape of a declare command and the structured event it should emit, so that everything downstream — paging, the channel, the timeline — keys off one action.

```bash
# A one-line declare from chat kicks off the whole machine
/incident declare \
  --title "Checkout success rate degraded" \
  --severity sev2 \
  --service checkout \
  --commander @oncall-payments
```

```json
{
  "event": "incident.declared",
  "incident_id": "INC-2271",
  "title": "Checkout success rate degraded",
  "severity": "sev2",
  "service": "checkout",
  "commander": "oncall-payments",
  "declared_at": "2026-06-20T02:16:04Z",
  "channel": "#inc-2271-checkout",
  "comms_cadence_minutes": 60
}
```

Notice that the declare event carries `declared_at` and that a channel gets created with the incident id baked into the name. Those two facts are what make the later timeline trivial to assemble: every message in `#inc-2271-checkout` is timestamped and scoped to this one incident. You are building the postmortem's evidence trail from the very first action.

There is a cultural prerequisite for a healthy bias to declare: declaring must be *psychologically safe and praised, not penalized.* If declaring a false alarm gets you teased or, worse, blamed for "crying wolf," people will stop declaring, and you will be back to limping. The fix is the same blameless stance that makes postmortems work — you reward the instinct to declare, you never punish a good-faith false alarm, and you treat "we declared early and it was nothing" as a win, because it means your reflexes are healthy. The on-call culture that supports this is covered in [designing a humane on-call](/blog/software-development/site-reliability-engineering/designing-a-humane-on-call).

### The declare-early culture, made concrete

"Bias to declare" is easy to say and hard to live, because the social pressure runs exactly the wrong way. A junior engineer who declares at 2am and turns out to be wrong feels exposed; the same engineer who quietly limps along and *happens* to fix it before anyone notices feels like a hero. Left alone, every incentive pushes toward under-declaring, and the only way to counter it is to make over-declaring visibly, repeatedly safe. Three concrete practices do this better than any poster on the wall.

First, **anyone can declare.** The authority to declare an incident must not be gated behind seniority or behind "ask your manager first." The person closest to the signal — often the most junior on-call — is the person who should pull the trigger, and any process that makes them hesitate has already cost you minutes. The declare command should be runnable by every engineer who carries the pager, full stop.

Second, **separate declaring from severity.** A lot of the hesitation to declare is actually hesitation to *commit to a severity* — "what if I call it a Sev1 and it's really a Sev3?" The fix is to decouple the two acts. Declaring just says "this is an incident, let us coordinate." Severity is a second, immediately-revisable estimate. When people understand that declaring is free and severity is cheap to revise, the declaring reflex gets much faster.

Third, **count and celebrate the false alarms.** A team that never declares a false alarm is not disciplined; it is under-declaring, and it is one missed signal away from a long outage. A healthy declare culture produces a steady trickle of "declared, stood down in ten minutes, no harm done" incidents, and a good engineering manager treats that trickle as a leading indicator of health, the same way a fire department treats a few false alarms as the unavoidable cost of a population that calls when it smells smoke. The metric to watch is not "how few false alarms" but "are we ever caught flat-footed by something we should have declared earlier."

The cost asymmetry is stark enough to put in a table. The point of laying it out this way is that the *expected* cost — probability times consequence — favors declaring in almost every realistic scenario, even when a false alarm is far more likely than a real incident.

| Scenario | What it costs you | How recoverable | Verdict |
| --- | --- | --- | --- |
| Declared, real incident | The response you needed anyway | Fully — this is the system working | Correct |
| Declared, false alarm | ~5–10 min, one stood-down IC, mild blush | Trivially — archive the channel | Cheap insurance |
| Did not declare, false alarm | Nothing | Nothing to recover | Lucky, not skilled |
| Did not declare, real incident | Uncoordinated sprawl, no IC, no timeline, growing blast radius | Painful — re-declare late, reconstruct events from memory | The disaster you are avoiding |

Look at the bottom two rows together. The only world in which *not* declaring pays off is the world where the thing was never an incident — and you cannot know that in advance, which is the entire point. Trading a guaranteed small cost (the false-alarm row) against a low-probability catastrophic cost (the under-declared real incident) is exactly the kind of bet where you buy the insurance every time. That is the arithmetic the bias to declare encodes.

## 4. Triage: turn "how bad is this?" into a severity

Once you have declared, the very next thing the IC does is triage. Triage answers exactly one question — *how bad is this?* — and it produces exactly one artifact: a **severity** level. The reason triage comes this early, before mitigation, is that severity *routes the entire response.* It decides who gets paged, whether an IC is mandatory, whether executives are notified, how often you communicate, and what response-time SLA you are holding yourself to. Get severity right and the rest of the response configures itself. Get it wrong and you either over-respond to a typo (waking up executives for a cosmetic bug) or, far more dangerously, under-respond to a fire (treating a spreading data-corruption bug like a minor annoyance).

The key idea — the thing that makes triage fast and calm instead of an argument — is that severity must be defined by a **written scale**, in advance, in terms of *user impact and scope*, not in terms of how stressed the responder feels. Two axes do almost all the work: **how many users are affected** (scope) and **how badly** (impact, with data loss and irreversibility as the most severe). With those two axes written down, severity becomes a lookup, not a debate.

![A matrix table mapping four severity levels to user impact, who responds, communication cadence, and response SLA](/imgs/blogs/the-anatomy-of-an-incident-3.png)

Here is a concrete, four-level scale you can adopt and adapt. The numbers (cadence, SLA) are illustrative starting points — tune them to your business — but the *structure* is the part to keep.

| Severity | User impact + scope | Who responds | IC required? | Exec notified? | Comms cadence | Response SLA |
| --- | --- | --- | --- | --- | --- | --- |
| **Sev1** | Major outage, data loss, or many users blocked. Core flow down. | On-call + IC + relevant teams, all hands | Yes | Yes, immediately | Public status page, every 30 min | Ack 5 min, bridge now |
| **Sev2** | Significant degradation or a key flow broken for a subset of users. | On-call + IC | Yes | Notify, not paged | Internal stakeholders, every 60 min | Ack 15 min |
| **Sev3** | Minor or partial issue, few users, a workaround exists. | On-call only | No | No | Ticket/thread updates as needed | Next business hour |
| **Sev4** | Cosmetic or no real user impact. Polish, typo, low-priority bug. | Backlog owner, async | No | No | None | Sprint queue |

Read that table as a routing function. The "who responds," "comms cadence," and "response SLA" columns are not suggestions — they are the automatic consequences of the severity you pick. That is the whole value: in the moment, the IC does not have to decide *whether* to notify executives or *how often* to update the status page. They decide the severity, and the table decides everything else. This is exactly the kind of decision you want to make once, calmly, in a planning meeting — and never again at 2am.

Because the two axes are scope and reversibility, you can capture triage as a tiny decision tree the on-call can run in fifteen seconds.

![A decision tree that routes a new incident to a starting severity based on scope and reversibility with a path to re-assess upward](/imgs/blogs/the-anatomy-of-an-incident-4.png)

The decision tree has one feature that matters more than all the others: the *re-assess upward* path. Severity is not a one-time stamp. It is an estimate you revise as you learn more, and there is a strong directional bias — **err toward the higher severity, and re-assess up the moment new evidence arrives.** The reason is the same cost asymmetry as declaring. Downgrading a Sev1 to a Sev2 because it turned out to be smaller than feared is cheap and painless. Upgrading a Sev3 to a Sev1 *after* it has been silently spreading for three hours is a disaster, because the response was sized for "minor" the entire time the problem was actually major. We will see exactly this failure mode in the second worked example.

### What each severity actually triggers

The table compresses a lot of consequence into a few columns, and it is worth unpacking each severity into the concrete machinery it sets in motion, because "Sev1" is only useful if everyone agrees on exactly what a Sev1 *does*. The whole value of the scale is that the level is a single word that expands into a fully specified response, so let us specify it.

A **Sev1** is the all-hands event. Declaring Sev1 should page not just the on-call but the incident commander on rotation, the relevant service owners, and — through an automated hook — open a bridge (a voice or video call) immediately, because at Sev1 the coordination cost is high enough that text alone is too slow. A Sev1 notifies leadership the moment it is declared, not after someone remembers to. It posts to the public status page on a fixed cadence, typically every 30 minutes, *even when there is no news* — "we are still investigating" on schedule is what keeps customers from flooding support, and a missed status update during a Sev1 reads to the outside world like the company has gone dark. A Sev1 also implies a hard response SLA: acknowledged within five minutes, a human actively working it within ten, and an IC who is doing nothing but running the response.

A **Sev2** is the serious-but-contained event. It pages the on-call and the IC but does not wake the whole org. It notifies internal stakeholders rather than broadcasting to the public status page, because the scope is a subset of users or a single non-core flow. The comms cadence relaxes to roughly every 60 minutes, and the response SLA loosens to acknowledged within fifteen minutes. The IC role is still mandatory at Sev2 — this is the level people most often get wrong, dropping the IC because "it's only a Sev2" and then discovering that nobody owned the response when it started to slip.

A **Sev3** is the on-call's own to handle. No IC, no executive notification, no status page. The on-call works it during business hours or at a relaxed pace, posts updates in a ticket or thread as there is something to say, and there is no fixed cadence because there are no anxious stakeholders to keep informed. The critical discipline at Sev3 is the *upgrade path*: a Sev3 that starts growing must be re-assessed upward without friction, which is exactly the failure the war story later in this post turns on.

A **Sev4** triggers essentially nothing in real time. It is a cosmetic or no-impact issue that goes to the backlog and gets fixed in the normal sprint flow. The reason it lives on the severity scale at all is so that the on-call has a *place to put* the things that are genuinely trivial, instead of either ignoring them (and losing track) or over-responding to them (and burning out). A clean Sev4 lane is what lets the on-call confidently *not* wake anyone for a broken footer image.

The deeper principle here is that severity is the **single routing key for the entire response**, and a routing key only works if it is unambiguous and if everything downstream genuinely keys off it. If your paging tool routes off severity but your status-page automation is a manual decision, you have two sources of truth and they will disagree at the worst possible moment. The mature setup wires *all* of the downstream consequences — paging, bridge creation, stakeholder notification, status-page cadence, SLA timers — off the one severity field on the incident, so that changing severity changes everything at once. That is what turns the table from documentation into mechanism.

#### Worked example: the severity routes the response, not the responder

Two pages arrive within a minute of each other on the same on-call. The first: the marketing site's footer is rendering a broken image on Safari. Scope: cosmetic, no flow affected. Reversibility: trivial. Lookup: **Sev4.** It goes to the backlog, the on-call does not wake anyone, and nobody updates a status page. The second: the login service is rejecting 100% of authentication for all users. Scope: everyone. Impact: core flow fully down. Lookup: **Sev1.** The on-call declares, pages the IC and the auth team, leadership is notified within minutes, and a public status page update goes out within thirty. Same on-call, same minute, wildly different responses — and the on-call did not have to *decide* on the response, only the severity. That is what a written scale buys you: it removes the in-the-moment judgment call from the most stressful moment, and replaces it with a lookup.

## 5. Mitigate: stop the bleeding before you understand it

Here is the stage that, more than any other, separates senior incident responders from junior ones. The instinct of a good engineer is to *understand* a problem before acting on it — that is exactly the right instinct when you are writing code or reviewing a design. During an incident it is the wrong instinct, and following it costs users dearly. The rule is blunt: **mitigate first, diagnose later.**

Mitigation is any action that restores service to users, *whether or not you understand why it works.* The canonical mitigations are reversals and escapes: roll back the deploy that started the trouble, fail over to a healthy region or replica, flip the feature flag off, shed load to protect the core path, drain the bad node, scale out the saturated tier. None of these require you to know the root cause. A rollback fixes a bad deploy whether the bug was a null-pointer, a config typo, or a slow query — you do not need to know which to roll back.

The principle is a statement about *who feels what.* Users feel the bleeding — the errors, the latency, the "please try again." They do not feel, and do not care about, the root cause. So the metric that maps to user pain is **time to mitigate (MTTM)**, the gap from "a human owns this" to "the bleeding has stopped." Time to *resolve* — the gap to a fully healed, root-caused, permanently-fixed system — matters to you, but it matters far less to the user, who stopped hurting the moment you mitigated. This is the reasoning behind treating mitigation as a hard priority over diagnosis: a one-line rollback that stops the bleeding in eight minutes beats a brilliant root-cause analysis that takes ninety minutes while users keep failing. You can — and should — diagnose afterward, on your own schedule, with the users safe. The full argument, with the failure modes of getting it backward, lives in [mitigate first, diagnose later](/blog/software-development/site-reliability-engineering/mitigate-first-diagnose-later); here I want you to internalize the ordering and the reason for it.

There is one honest caveat, and it is important: *mitigate first* assumes your mitigation is safe and reversible. A blind rollback is usually safe. But some mitigations can make things worse — failing over to a replica that is also corrupted, or "fixing" a data-corruption incident by restarting the service that is still actively writing bad rows. The discipline is: prefer the safest mitigation that stops the bleeding, and if the only available mitigation is risky, that risk goes through the IC, not through a lone engineer's keyboard. When the failure mode is data corruption rather than availability, sometimes the correct first mitigation is to *halt writes* — stop making it worse — before you do anything else. We will see exactly that in the second worked example.

Mitigations should be pre-built and one command away. The two most common — rollback and feature-flag kill — should never require improvisation at 2am. Here is what "one command away" looks like for a Kubernetes rollback and for a kill switch.

```bash
# Mitigation #1: roll back the last deploy. No root cause needed.
kubectl rollout undo deployment/checkout -n payments
kubectl rollout status deployment/checkout -n payments --timeout=120s

# Mitigation #2: kill the risky feature behind a flag, instantly
curl -XPOST https://flags.internal/api/v1/flags/new_checkout_flow \
  -H "Authorization: Bearer $FLAG_TOKEN" \
  -d '{"enabled": false, "reason": "INC-2271 mitigation"}'
```

The design lesson behind those two commands is *make mitigation cheap to reach for.* If your only path to recovery is a forty-minute redeploy, you have a forty-minute MTTM floor no matter how fast you detect and declare. Teams that take MTTM seriously invest in fast, safe rollbacks, feature flags on anything risky, and pre-tested failover — so that when the moment comes, the mitigation is a single command the on-call already knows. The architectural patterns that *enable* fast mitigation (circuit breakers, bulkheads, graceful degradation, load shedding) are designed in, not added during the incident; for that design layer, cross over to system-design's treatment of [cascading failures, circuit breakers, and bulkheads](/blog/software-development/system-design/cascading-failures-circuit-breakers-and-bulkheads) and [reliability, SLOs, error budgets, and graceful degradation](/blog/software-development/system-design/reliability-slos-error-budgets-and-graceful-degradation).

### Choosing a mitigation: speed, blast radius, and reversibility

Not all mitigations are equal, and during an incident you are choosing under time pressure between several that *could* stop the bleeding. The decision is a three-way trade among how fast the mitigation takes effect, how reversible it is if it makes things worse, and how much of the system it touches. A senior responder has an internal ranking of these and reaches for the safest fast option first. Laying that ranking out as a table makes the reasoning teachable rather than something you only acquire after your third bad night.

| Mitigation | Typical time to effect | Reversibility | Blast radius | When to reach for it |
| --- | --- | --- | --- | --- |
| Feature-flag kill | Seconds | Trivial — flip it back on | Just the flagged feature | A specific new feature is the suspect; flag exists |
| Roll back the deploy | 1–3 min | Easy — roll forward again | The whole service | A recent deploy correlates with onset |
| Fail over region/replica | 1–5 min | Moderate — fail back | Whole service, new dependencies | The primary is unhealthy and a healthy standby exists |
| Shed load / rate-limit | Seconds | Easy — restore limits | Some users rejected on purpose | Saturation or a cascade; protect the core path |
| Scale out the tier | 2–10 min | Easy — scale back | The scaled tier | Pure capacity shortfall, no bad code |
| Halt writes | Seconds | Hard — backlog to drain | Writers blocked | Data corruption that is actively spreading |

Read the table top to bottom as a rough priority order for an *availability* incident: a feature-flag kill is the dream mitigation because it is instant, perfectly reversible, and surgically narrow, so if a flag exists for the suspect change you reach for it first. A rollback is the workhorse — slightly slower and broader, but still easy to undo. Failover is powerful but carries more risk because it changes which dependencies you lean on, and a failover to a standby that shares the failing dependency buys you nothing. The bottom row, halting writes, is the odd one out: it is the *worst* mitigation for an availability problem (you are deliberately rejecting work) but the *correct first* mitigation for a spreading-corruption problem, because the priority there is "stop making it worse" before "restore service." Matching the mitigation to the *shape* of the failure — availability versus correctness — is the judgment the table is meant to build.

The honest caveat from earlier applies most sharply to the riskier rows. A feature-flag kill or a rollback you can essentially always do on your own authority, because they are reversible and narrow. A failover or a write-halt has consequences that outlast the incident, so those decisions route through the IC, who can weigh them against the bigger picture and who owns the call if it goes wrong. The rule is not "the on-call cannot act" — it is "the more irreversible the mitigation, the higher up the decision goes," which keeps a lone tired engineer from making a region-wide bet at 2am without a second set of eyes.

## 6. Diagnose, then resolve: the all-clear is a checklist, not a feeling

Only after the bleeding has stopped do you switch into diagnosis. The relief of mitigation is exactly what gives you the room to diagnose well — carefully, with the system safe, without the pressure of a climbing error rate. Diagnosis is where the deep-debugging skills come in: form a hypothesis, look at the evidence, test it, narrow it down. This is the scientific method applied to production, and it is worth borrowing the rigor from the debugging series' [stop guessing — the scientific method of debugging](/blog/software-development/debugging/stop-guessing-the-scientific-method-of-debugging). The key mindset is that diagnosis is *not* on the critical path for user recovery anymore — users are already safe — so you optimize diagnosis for *correctness*, not speed.

A subtle but important point: mitigation and diagnosis can overlap in time. While one engineer rolls back (mitigation), another can be pulling logs and traces to understand the cause (diagnosis). What must not happen is *diagnosis blocking mitigation* — nobody should be saying "wait, don't roll back yet, I want to understand it first" while users are failing. Understand it after. The rollback does not destroy the evidence; the logs, traces, and metrics are still there.

Resolution is the final stage, and it has its own discipline: an **all-clear is a checklist, not a feeling.** The single most common way incidents go wrong at the end is a premature all-clear — someone sees the error rate drop for thirty seconds, says "looks good, we're resolved," closes the channel, and goes back to bed. Then the problem relapses at 03:30 and the team has to re-declare, re-page, and re-mitigate, except now they are more tired and the timeline has a confusing gap. To resolve correctly you check explicit exit criteria. A solid set:

- The SLI has recovered to its normal baseline (not just "less bad" — *normal*).
- It has stayed recovered for a defined monitoring window — typically two consecutive clean evaluation windows, so a single lucky scrape cannot fool you.
- No dependent service is still degraded as a side effect.
- The mitigation is stable, not a fragile hold (e.g., you are not one restart away from the problem returning).
- The timeline is captured and the postmortem owner is assigned.

Because incidents can relapse, it is most honest to model an incident as a small **state machine** with explicit entry and exit conditions for each state — including a `reopened` state for relapses. This is what keeps a team from declaring victory by vibes.

![A state machine diagram showing an incident moving from declared through investigating, mitigated, and monitoring to resolved with a reopened path for relapses](/imgs/blogs/the-anatomy-of-an-incident-5.png)

The states map cleanly onto the lifecycle: `DECLARED` (you have triaged and assigned severity), `INVESTIGATING` (triage and mitigation in progress, user impact understood), `MITIGATED` (bleeding stopped, impact flat or trending down), `MONITORING` (watching for relapse, waiting for the clean windows), `RESOLVED` (exit criteria met, MTTR recorded), and `REOPENED` (a relapse, which sends you back to re-mitigate and may bump the severity). The value of writing it as a state machine is that the transition from `MONITORING` to `RESOLVED` becomes a *checked condition* — clean for N windows, dependents healthy, mitigation stable — rather than a tired engineer's hope. That single discipline kills the premature all-clear.

### Entry and exit criteria for each incident state

The reason a state machine is more than a diagram is that each transition has an *explicit gate* — a condition that must be true to move on. When those gates are written down, the room never argues about whether it is time to advance; it checks the gate. Here are the gates that make the machine real.

`DECLARED` is *entered* the moment a human declares, and it carries an initial severity. It is *exited* into `INVESTIGATING` as soon as someone starts actively working the problem — usually within the same minute. The only thing that must be true to leave `DECLARED` is that an owner exists; if there is no owner, you are still limping, not investigating.

`INVESTIGATING` is *entered* when work begins and *exited* into `MITIGATED` when, and only when, user-facing impact has stopped or is provably trending to zero. The gate here is the most important one in the whole machine: you do not get to claim `MITIGATED` because you *think* the fix worked — you claim it when the SLI shows the bleeding has stopped. This is where mitigation and diagnosis run in parallel: one engineer drives toward the `MITIGATED` gate while another begins the diagnosis that will eventually let you exit `MONITORING` for good.

`MITIGATED` is *entered* on confirmed impact reduction and *exited* into `MONITORING` immediately — the two are almost the same instant — but the distinction matters because `MITIGATED` records the MTTM timestamp. The moment you cross this gate is the moment users stopped hurting, and it is the single most valuable timestamp in the timeline.

`MONITORING` is the gate that prevents the premature all-clear, and it is the only state with a *time-based* exit condition. You *enter* it the moment the bleeding stops, and you *exit* it into `RESOLVED` only when every one of the resolution exit criteria is true: the SLI has held at its normal baseline for N consecutive clean evaluation windows, no dependent service is still degraded, the mitigation is stable rather than a fragile hold, and a postmortem owner has been assigned. Notice that this gate cannot be satisfied by a single good measurement — it requires *duration*, which is exactly what defeats the thirty-second-recovery trap.

`RESOLVED` is terminal in the happy path. You *enter* it when the `MONITORING` gate clears, you record MTTR, and you hand off to the learn stage. The only way out of `RESOLVED` is backward, into `REOPENED`, if the problem relapses after you thought you were done.

`REOPENED` is *entered* on any relapse of the original symptoms and routes you straight back to re-mitigate. Crucially, reopening the *same* incident rather than declaring a fresh one preserves the continuity of the timeline — the relapse, the re-mitigation, and the eventual real fix all live in one record, which is what lets the postmortem see the full arc instead of two disconnected fragments. A relapse is also a strong signal to *re-assess severity upward*, because a problem that comes back is, by definition, not as understood or as fixed as you believed.

The thing to take from writing the gates out is that an incident never advances on a feeling. Every transition is a checked condition, and the checked condition for leaving `MONITORING` — duration plus a clean dependents check plus a stable mitigation — is the one that earns its keep on almost every incident, because the urge to close early and go back to bed is relentless and the gate is the only thing that reliably resists it.

#### Worked example: the premature all-clear that cost an hour

A database connection pool exhausts during a traffic spike; the on-call bumps the pool size (mitigation), the error rate drops within ninety seconds, and they immediately mark the incident resolved and close the channel. Twenty minutes later, a second spike exhausts the *new* larger pool too, errors return, and they have to re-declare INC-2271-B, re-page, and re-investigate — now realizing the real fix was a connection leak, not pool size. Had they used the state machine, the incident would have sat in `MONITORING` through that second spike, the relapse would have been caught inside the same incident, and the diagnosis would have reached the leak the first time. The premature all-clear did not save time; it cost an extra page, an extra hour, and a confused timeline. The exit criteria exist precisely to prevent this.

## 7. The four "mean time to" metrics, and why mitigate-time is the one that matters

Every well-run incident produces a *timeline* — a timestamped record of detect, ack, declare, mitigate, diagnose, resolve. That timeline is not paperwork; it is the raw material for the metrics that tell you whether your incident response is getting better or worse. There are four "mean time to" metrics, and the most common mistake is to track only the last one (MTTR) and miss the one that actually maps to user pain (MTTM).

![A layered stack of the four mean-time metrics showing time to detect, acknowledge, mitigate, and resolve as separate gaps in the timeline](/imgs/blogs/the-anatomy-of-an-incident-6.png)

Each metric measures a *different gap* in the timeline, and that is the whole point — they decompose recovery time into pieces you can attack separately.

- **MTTD — mean time to detect.** Problem starts → a human (or alert) knows. Shrink it with better SLIs and symptom-based alerting. This is your detection quality.
- **MTTA — mean time to acknowledge.** Alert fires → a human acknowledges and owns it. Shrink it with humane on-call: reachable pagers, sane escalation, an on-call who is not buried in noise. A high MTTA usually means alert fatigue.
- **MTTM — mean time to mitigate.** A human owns it → the bleeding has stopped. This is the number users actually feel, because it is the duration of their pain. Shrink it with pre-built, one-command mitigations: fast rollback, feature flags, tested failover.
- **MTTR — mean time to resolve.** A human owns it (or the problem starts, depending on your definition) → the system is fully healed and root-caused. This feeds the postmortem and trends your overall response health.

A small but real caution on definitions: organizations define the *start* of MTTR differently — some measure from problem onset, some from detection, some from acknowledgment. The letter "R" is also overloaded; "MTTR" sometimes means *repair*, sometimes *recovery*, sometimes *resolve*. None of this matters as long as you **pick one definition and apply it consistently**, so your trend line is comparing like with like. Be explicit in your runbook about exactly which timestamps bound each metric.

Now the punchline, and it is a claim worth defending: **for users, MTTM matters more than MTTR.** Consider an incident that is mitigated by a rollback in 10 minutes but not fully resolved (root cause found, permanent fix shipped, all monitoring green) for 4 hours. The user experience of that incident is *ten minutes long* — after the rollback, users are fine. The remaining 3 hours and 50 minutes is your team doing careful diagnosis and a clean fix with the users safe. If you reported only MTTR, you would record a "4-hour incident" and might panic about it; if you track MTTM, you correctly see a 10-minute user-facing impact and a long-but-calm cleanup. Optimizing the wrong metric leads to the wrong investments — you would pour effort into faster root-causing (which users do not feel) instead of faster mitigation (which they do).

You can compute these straight from your incident records. Here is a small Python sketch that takes a timeline and produces all four metrics, which you would run across your incident history to trend them.

```python
from datetime import datetime

def t(s):  # parse ISO-8601 timestamps from the incident record
    return datetime.fromisoformat(s)

def incident_metrics(tl):
    problem_start = t(tl["problem_started_at"])
    detected      = t(tl["detected_at"])
    acked         = t(tl["acknowledged_at"])
    mitigated     = t(tl["mitigated_at"])
    resolved      = t(tl["resolved_at"])

    mins = lambda a, b: round((b - a).total_seconds() / 60, 1)
    return {
        "MTTD_min": mins(problem_start, detected),   # detection quality
        "MTTA_min": mins(detected, acked),           # on-call health
        "MTTM_min": mins(acked, mitigated),          # the number users feel
        "MTTR_min": mins(problem_start, resolved),   # overall, feeds postmortem
    }

inc = {
    "problem_started_at": "2026-06-20T02:14:00",
    "detected_at":        "2026-06-20T02:14:00",
    "acknowledged_at":    "2026-06-20T02:16:00",
    "mitigated_at":       "2026-06-20T02:22:00",
    "resolved_at":        "2026-06-20T02:51:00",
}
print(incident_metrics(inc))
# {'MTTD_min': 0.0, 'MTTA_min': 2.0, 'MTTM_min': 6.0, 'MTTR_min': 37.0}
```

The reason to compute these per-incident and trend the medians (not the means — a single 6-hour outlier wrecks a mean) is that each metric points at a *different fix*. A rising MTTA says "your on-call is buried, fix the alert noise." A rising MTTM says "your mitigations are too slow, invest in rollback and flags." A rising MTTD says "your detection is missing things, fix your SLIs." Lumping it all into one MTTR number hides which lever to pull. For how to actually build the dashboards that surface these honestly, see [dashboards that tell the truth](/blog/software-development/site-reliability-engineering/dashboards-that-tell-the-truth).

The choice of median over mean is not a statistical nicety — it is a defense against being misled by your own data. Incident durations are heavily right-skewed: most incidents are short, and a handful are enormous. A single multi-hour regional outage can drag the *mean* MTTR up by an hour even if your typical incident got dramatically faster, which means a team that genuinely improved can look like it regressed, and a team that got worse on the common case can hide behind the absence of a big outage. The median ignores the size of the tail and reports what a *typical* incident looked like, which is the number you actually want to drive down. If you care about the tail too — and you should, because the long outages are where the real budget gets spent — track a high percentile like p90 alongside the median, but never let the mean be your headline number. The same discipline that makes you prefer p99 latency over average latency applies here: averages launder the worst experiences into a comfortable middle that nobody actually had.

There is also a sequencing insight buried in the four metrics that is easy to miss. They are *additive* — they tile the timeline end to end with no gaps — which means your total time-to-resolve is the sum of the four gaps, and you can attack the largest gap first for the biggest win. If your median timeline is MTTD 8 minutes, MTTA 1 minute, MTTM 4 minutes, and the long tail of resolution after that, then your detection is the fat part and your alerting investment should go there, not into shaving the already-tight 4-minute mitigation. Decomposing recovery into these four additive gaps turns "make incidents shorter" — a vague aspiration — into "shrink the biggest gap" — a concrete, measurable project. That is the entire reason to bother measuring four numbers instead of one.

#### Worked example: a Sev1 timeline, end to end with numbers

Walk a realistic Sev1 to see the metrics fall out of the timeline. At 14:02 a config push to the API gateway starts rejecting a growing fraction of authenticated requests; by 14:05 it is 100% of logins failing. The symptom alert on login success fires at 14:06 (problem onset was 14:02, so MTTD is 4 minutes — the config took a couple of minutes to fully propagate before the SLI crossed the threshold). The on-call acknowledges at 14:07 (MTTA 1 minute) and declares Sev1 immediately, which auto-opens a bridge and pages the IC and the platform team. Triage at 14:09 confirms the scope — every user, core flow fully down — and the severity holds at Sev1. The IC asks for the fastest reversible mitigation; the platform engineer rolls back the gateway config at 14:14, and login success climbs back to baseline by 14:16, so MTTM is 9 minutes measured from the 14:07 acknowledgment. The incident sits in monitoring while diagnosis pins the cause on a malformed rate-limit rule in the pushed config, and after two clean windows and a dependents check the IC calls the all-clear at 14:41. MTTR, measured from the 14:02 onset, is 39 minutes.

Now read the budget impact, because this is where a Sev1 stings. Login was fully down — 100% impact — for the window from roughly 14:05 (full failure) to 14:16 (recovered), about 11 minutes of total-user-facing downtime. Against a 99.9% monthly SLO with its 43.2-minute budget, this single incident spent about 11 of the 43.2 minutes, or roughly a *quarter of the entire month's budget in one push*. Contrast that with the Sev2 checkout incident earlier, which touched 8% of users for 8 minutes and spent well under a minute of equivalent budget. The arithmetic makes the severity scale's instinct concrete: the same eleven minutes of clock time costs you fourteen times more budget at 100% impact than at a partial scope, which is precisely why a 100%-impact core-flow failure is a Sev1 and a partial-scope degradation is a Sev2. The budget is doing the routing the human intuition was already reaching for.

The other lesson in that timeline is how much the *mitigation* did versus the *resolution*. Users were down for 11 minutes and fine thereafter; the remaining 25 minutes from 14:16 to 14:41 was the team carefully confirming the fix and watching for relapse, with zero additional user pain. If you reported only the 39-minute MTTR, you would tell the story of a 39-minute outage. The truth the MTTM tells is an 11-minute outage followed by a calm, safe cleanup — and that distinction is the difference between an org that panics and over-rotates on resolve-time and an org that correctly invests in the rollback path that made the 9-minute mitigation possible.

#### Worked example: 99.9% SLO and the budget an incident spends

Connect the timeline to the error budget — the currency of the whole series. A 99.9% monthly availability SLO gives you a budget of 0.1% of the month as allowed downtime. With about 43,200 minutes in a 30-day month, that is roughly $43{,}200 \times 0.001 = 43.2$ minutes of budget per month. Our Sev2 example caused user-facing impact from 02:14 (problem start) to 02:22 (mitigated) — about 8 minutes of full-severity impact, but only 8% of users were hit, so the *budget* spent is closer to $8 \times 0.08 \approx 0.64$ minutes of equivalent total-downtime budget. That single incident spent under 2% of the month's 43.2-minute budget. Now flip it: the *premature-all-clear* relapse from section 6 would have doubled the user-facing window, and a Sev1 that takes 40 minutes to mitigate at 100% impact would blow the *entire* month's budget in one night. This is the link between incident response and the error budget: every minute of MTTM is a withdrawal from the budget. Tighten MTTM and you preserve budget; preserve budget and you keep shipping. The mechanics of the budget itself are in [the error budget, the currency of reliability](/blog/software-development/site-reliability-engineering/the-error-budget-the-currency-of-reliability).

## 8. The incident timeline: the artifact that calms the room and feeds the postmortem

The timeline deserves its own section because it does double duty. *During* the incident, a running timeline is what keeps the room calm — it answers "what do we know and what have we tried" without anyone having to re-explain, and it lets a fresh responder get up to speed in thirty seconds. *After* the incident, the same timeline is the spine of the postmortem and the source of every metric we just computed. The discipline is simple: **someone writes down every significant action, with a timestamp, as it happens.** On a small incident the IC can do this; on a big one, you assign a dedicated scribe.

![A timeline of a Sev2 incident with timestamped events from alert at 02:14 through declare, triage, mitigate, diagnose, and resolve at 02:51](/imgs/blogs/the-anatomy-of-an-incident-7.png)

The timeline above is our worked Sev2 in full: the alert fires at 02:14, the on-call acknowledges and declares at 02:16, triage at 02:18 establishes that 8% of users are hitting checkout errors, a rollback at 02:22 stops the bleeding, diagnosis at 02:35 pins it on a bad migration, and the all-clear comes at 02:51 once the SLI has held normal for two clean windows. Read it as the four metrics made visible: the gap to 02:16 is MTTA, the gap to 02:22 is MTTM, the gap to 02:51 is MTTR. A timeline like this is worth a thousand words of recollection, because human memory of a stressful 2am is unreliable — people misremember the order of events, who decided what, and how long things took. The timestamped record is the truth.

Here is a timeline template you can drop into the incident channel's pinned message and fill in live. Keeping it structured means it parses cleanly into your metrics later.

```yaml
incident: INC-2271
title: "Checkout success rate degraded"
severity: sev2          # re-assess and update if scope changes
commander: "@oncall-payments"
scribe: "@eng-amir"
service: checkout

timeline:
  - at: "02:14"  stage: detect    note: "Alert CheckoutSuccessRateLow fired (97.1%)"
  - at: "02:16"  stage: declare   note: "Declared Sev2, IC named, #inc-2271 opened. MTTA 2m"
  - at: "02:18"  stage: triage    note: "Scope: ~8% of users, checkout only. Confirmed Sev2"
  - at: "02:22"  stage: mitigate  note: "Rolled back deploy v412 -> v411. Errors falling. MTTM 6m"
  - at: "02:28"  stage: mitigate  note: "SLI back to 99.6%. Entering monitoring"
  - at: "02:35"  stage: diagnose  note: "Root cause: migration added a non-indexed WHERE; slow query storm"
  - at: "02:49"  stage: monitor   note: "Two clean 5m windows. Dependents healthy"
  - at: "02:51"  stage: resolve   note: "All-clear. MTTR 37m. Postmortem owner @eng-priya"

metrics:
  mttd_min: 0
  mtta_min: 2
  mttm_min: 6
  mttr_min: 37
budget_spent_min_equiv: 0.6   # ~8 min impact x ~8% of users
postmortem: "due within 5 business days"
```

Two practices make the timeline reliable. First, *write actions as you take them*, not from memory afterward — "rolling back now" goes in the channel before you hit enter on the rollback, so the timestamp is the truth. Second, *note decisions, not just facts* — "decided to roll back rather than hotfix because rollback is faster and reversible" is the kind of context the postmortem desperately needs and nobody remembers a week later. The handoff to learning is then trivial: the postmortem author starts from a complete, timestamped, decision-annotated record instead of a foggy reconstruction.

### The scribe, and the timeline anti-patterns to avoid

On a small incident the IC can keep the timeline themselves, but on anything larger that is a mistake, because the IC's attention is the scarcest resource in the room and timeline-keeping is exactly the kind of work you can hand off. The fix is a dedicated **scribe** — one person whose entire job is to capture the running record while the IC coordinates and the responders fix. The scribe is not a junior-only chore; on a serious Sev1 it is one of the most valuable roles in the channel, because a good scribe lets the IC stay heads-up on the response instead of heads-down typing. The scribe captures three things: every action taken (with who and when), every decision made (with the reasoning), and every meaningful change in the system's state (the SLI crossed back over the line at 14:16). What the scribe deliberately does *not* do is editorialize or assign blame in the moment — the record is facts and decisions, not judgments, which keeps it usable as blameless postmortem material later.

Several timeline anti-patterns recur often enough to call out by name. The first is the **reconstruct-it-later timeline**, where nobody writes anything live and someone tries to rebuild the sequence from memory and scrollback the next morning. It is always wrong in the details that matter most — the order of events and how long each gap took — because stressed human memory compresses and reorders time. The second is the **facts-without-decisions timeline**, which records what happened but not *why* anyone chose to do it, so the postmortem can see that the team rolled back but cannot see that they debated a hotfix first and rejected it for a good reason; the most valuable line in any timeline is usually a decision, not an event. The third is the **everything-everywhere timeline**, where the channel fills with so much chatter that the actual signal — the real actions and decisions — drowns in side conversations, which is why a structured pinned timeline that the scribe curates beats trying to use the raw channel scrollback as your record. The fourth is the **clock-skew timeline**, where different people quote times from their own laptops and the sequence becomes internally inconsistent; the cure is to agree at declare time that all timestamps come from one source, ideally the incident tool's server clock, so the record is coherent.

The payoff for getting the timeline right is that it serves three masters at once without extra work. Live, it is the shared situational awareness that lets a responder who joins at 14:20 catch up in thirty seconds instead of interrupting the IC. At resolution, it is the source of the four metrics — every gap is computed from two timestamps already in the record. And afterward, it is the spine of the postmortem, the decision-annotated narrative that lets the team find the contributing factors without re-litigating a foggy memory of a stressful night. One artifact, written once, live, pays for itself three times. That return is why the discipline of writing as you act is worth enforcing even when, at 2am, the temptation is to just fix the thing and document it later.

## 9. War story and a severity mis-classification: when "Sev3" was really a Sev1

Two real-shaped stories, because the abstract lifecycle lands harder when you see it break.

**The Google error-budget model and the rise of the IC.** The modern incident lifecycle did not appear from nowhere. Google's SRE practice popularized two ideas that are now industry default: the *incident commander* role borrowed from emergency-services incident command systems, and the discipline of separating the *response* (run by the IC) from the *technical fix* (run by subject-matter experts). The insight is that during a large incident, coordination is a full-time job — deciding, delegating, and communicating cannot be done by the same person who has their head in the logs. The documented Google practice, written up in the SRE Book's chapter on managing incidents, is the canonical source for the roles and the bias toward declaring early. This is well-documented, not illustrative.

**The classic cascading failure.** The most-cited reason mitigation must beat diagnosis is the cascading-failure outage: a small trigger (a slow dependency, a bad config push, a thundering herd of retries) causes a tier to saturate, which causes upstream timeouts, which cause *more* retries, which saturate further — a feedback loop that takes down healthy services. In a cascade, the only thing that stops the bleeding fast enough is mitigation that breaks the loop: shed load, trip the breaker, drain traffic. Spending those minutes diagnosing the original trigger while the cascade spreads is how a 5-minute problem becomes a 2-hour multi-region outage. The architecture that prevents cascades is covered in [cascading failures, circuit breakers, and bulkheads](/blog/software-development/system-design/cascading-failures-circuit-breakers-and-bulkheads) and the real-postmortem case studies are in [anatomy of an outage, lessons from real postmortems](/blog/software-development/system-design/anatomy-of-an-outage-lessons-from-real-postmortems); this post is the *response* to such an outage, that post is the *anatomy of the outage itself.*

**The severity mis-classification (illustrative, but composed from real patterns).** Here is the one that teaches the err-high lesson. A background job that re-computes user account balances starts writing a small number of incorrect values after a deploy. The on-call sees "a few wrong balances, looks minor, there's a workaround," and files it as a **Sev3** — on-call only, no IC, no escalation, ticket-thread updates as needed. Severity stamped once, never re-assessed.

![A before and after comparison showing a misfiled Sev3 that let data corruption spread for hours versus an err-high approach that escalated to Sev1 and capped the blast radius](/imgs/blogs/the-anatomy-of-an-incident-8.png)

The trouble is that this was not an availability incident — it was a *data-correctness* incident, and data corruption has a property availability problems do not: **it spreads, and it is not self-limiting.** The bad balances fed downstream jobs, which computed more bad values, which fed reports and notifications. For three hours the on-call treated it as a Sev3 backlog item while the corruption silently expanded from a handful of rows to roughly 40,000. By the time a customer-facing report exposed the scale, the response had to escalate to Sev1 in a panic, halt the affected writes, and restore from backup — a far larger, riskier operation than it would have been three hours earlier.

The lesson is exactly the bias the severity tree encodes: **err toward the higher severity, and re-assess as you learn.** Two specific rules fall out of this story. First, *irreversibility outranks scope.* "Few users affected" suggested low severity, but "data corruption that spreads and cannot be cheaply undone" should have outranked the small initial scope and started this at Sev2-trending-Sev1. The right first mitigation for a spreading-corruption incident is often to *halt the writes* — stop making it worse — before anything else, which the Sev3 framing never prompted. Second, *severity is a live estimate, not a stamp.* Had the on-call re-assessed at the first sign of spread — "wait, the count of bad rows is growing" — they would have bumped to Sev1, pulled in an IC, and capped the blast radius at a few hundred rows instead of 40,000. The cost of starting too high (a needless IC page, quickly stood down) is trivial next to the cost of starting too low on an irreversible problem. When the impact is data loss or corruption, the backups and restore drills that save you are their own discipline — and a backup you have never restored is not a backup; that thread lives in the disaster-recovery posts of the series.

## 10. How to reach for this, and when not to

The lifecycle and the severity scale are not free. They are *process*, and process has a cost: ceremony, declared incidents that turn out to be nothing, time spent updating timelines. Like every SRE practice, you apply it where the reliability return justifies the cost, and you say plainly where it does not.

**Reach for the full lifecycle when** user-facing reliability is on the line — anything that touches a real SLO, anything customers feel, anything with data at risk. For those, the discipline pays for itself many times over: the calm, the routed response, the timeline, the metrics, the postmortem fuel. The bias to declare, the written severity scale, and the mitigate-first ordering are the three highest-leverage habits, and they cost almost nothing once they are muscle memory.

**Do not over-apply it.** A few honest "when not to":

- **Don't declare a formal incident for a Sev4.** A cosmetic bug with no user impact does not need a channel, an IC, and a cadence. File a ticket. Reserve the machinery for things that hurt users. Over-declaring trivia trains people to ignore declarations.
- **Don't run a Sev1 process for a Sev3.** Waking executives and posting public status updates for a minor, few-user issue with a workaround burns goodwill and trust. The severity scale exists to right-size the response in *both* directions.
- **Don't let process block mitigation.** If the building is on fire and you know the fix is a one-command rollback, roll back *now* and write the timeline entry a beat later. The lifecycle serves the recovery; it does not gate it. A team that refuses to mitigate until the IC has formally taken the bridge has misunderstood the priority order.
- **Don't demand a heavyweight timeline for a 3-minute Sev4.** Match the artifact's weight to the incident's severity. A one-line note is a fine "timeline" for a trivial issue.
- **Don't optimize MTTR while ignoring MTTM.** As section 7 argued, the user feels mitigate-time. A team proudly driving down resolve-time while mitigate-time stays high is polishing the wrong number.

The meta-rule: the lifecycle is a tool for staying calm and right-sizing the response, not a bureaucracy to satisfy. Scale the ceremony to the severity, always preserve the mitigate-first ordering, and keep the bias to declare.

## Key takeaways

- **An incident has an anatomy.** Detect, declare, triage, mitigate, diagnose, resolve, then learn — in that order. Knowing the stage you are in turns a panic into a checklist with one obvious next move.
- **Declaring is cheap; limping is expensive.** Bias hard toward declaring. A false-alarm declaration costs minutes; an under-declared real incident grows uncoordinated while users suffer.
- **Severity is a lookup, not an argument.** Define a written Sev1–Sev4 scale by user impact and scope, and let it route who responds, the comms cadence, and the SLA automatically.
- **Err toward the higher severity and re-assess as you learn.** Downgrading is cheap; upgrading a silently-spreading problem hours late is a disaster. Irreversibility (data corruption) outranks initial scope.
- **Mitigate first, diagnose later.** Users feel the bleeding, not the root cause. A one-command rollback that stops the bleeding beats a brilliant root-cause analysis that takes an hour while users fail.
- **The all-clear is a checklist, not a feeling.** Resolve only against explicit exit criteria — SLI back to baseline, stable for N clean windows, dependents healthy — to kill the premature all-clear and the relapse.
- **Track four metrics, and weight MTTM.** MTTD, MTTA, MTTM, MTTR each point at a different fix. The one users feel is mitigate-time; optimizing only resolve-time polishes the wrong number.
- **The timeline is the artifact.** A timestamped, decision-annotated record calms the room live and feeds both the metrics and the postmortem afterward. Write it as you act, not from memory.

## Further reading

- [Reliability is a feature: the SRE mindset](/blog/software-development/site-reliability-engineering/reliability-is-a-feature-the-sre-mindset) — the series intro and the define→measure→budget→respond→learn loop this post sits inside.
- [Mitigate first, diagnose later](/blog/software-development/site-reliability-engineering/mitigate-first-diagnose-later) — the full argument for the ordering, with the failure modes of getting it backward.
- [Incident command: staying calm under fire](/blog/software-development/site-reliability-engineering/incident-command-staying-calm-under-fire) — the IC role, the bridge, and the comms discipline that the declare stage triggers.
- [Designing a humane on-call](/blog/software-development/site-reliability-engineering/designing-a-humane-on-call) — the on-call culture and escalation that keep MTTA low and make the bias to declare safe.
- [The blameless postmortem](/blog/software-development/site-reliability-engineering/the-blameless-postmortem) — the learn stage this post hands off to, and why blameless surfaces more truth.
- [The error budget: the currency of reliability](/blog/software-development/site-reliability-engineering/the-error-budget-the-currency-of-reliability) — how every minute of MTTM is a withdrawal from the budget.
- [Anatomy of an outage: lessons from real postmortems](/blog/software-development/system-design/anatomy-of-an-outage-lessons-from-real-postmortems) — the system-design companion: the anatomy of the outage itself, versus this post's anatomy of the response.
- [Cascading failures, circuit breakers, and bulkheads](/blog/software-development/system-design/cascading-failures-circuit-breakers-and-bulkheads) — the architecture that makes fast mitigation possible.
- The Google SRE Book, "Managing Incidents" and "Emergency Response" chapters — the canonical source for the incident commander role and the bias to declare early.
