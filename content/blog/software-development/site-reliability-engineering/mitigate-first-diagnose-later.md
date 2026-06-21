---
title: "Mitigate First, Diagnose Later: The Counterintuitive Skill That Stops the Bleeding"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Learn the single most counterintuitive incident skill — stop the bleeding before you understand why it's bleeding — with a mitigation toolkit, a lever decision table, and two worked outages."
tags:
  [
    "site-reliability-engineering",
    "sre",
    "incident-response",
    "mitigation",
    "rollback",
    "feature-flags",
    "mttr",
    "on-call",
    "reliability",
  ]
category: "software-development"
subcategory: "Site Reliability Engineering"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/mitigate-first-diagnose-later-1.png"
---

It's 14:04 on a Tuesday. The pager goes off. The checkout service is throwing errors on 8% of requests, p99 latency has tripled, and the burn-rate alert says you are torching a month of error budget every few hours. Three engineers join the bridge. Within ninety seconds, somebody is already deep in a log query, another is tailing a trace, and a third is asking, "Wait, what changed? Did anyone touch the cart service?" Everyone leans in. The instinct in the room — the deep, almost physical instinct of every good engineer — is to *understand* the problem before touching it. To find the bug. To explain it. To be right.

That instinct is correct in your IDE on a calm afternoon. Under fire, it is wrong, and it is expensive. Every minute the team spends root-causing is a minute users cannot check out. The outage does not pause politely while you read logs. If you can roll back a deploy in four minutes but a live diagnosis will take forty, then choosing to diagnose first is choosing to keep users down for thirty-six extra minutes — for the sake of your curiosity. The job during an incident is not to be right. The job is to make the pain stop. Understanding can — and should — wait.

This is the single most counterintuitive skill in incident response, and it has a name: **mitigate first, diagnose later**. Mitigation means restoring service. Diagnosis means finding the cause. These are *separate goals*, and the discipline is to recognize that mitigation comes first and stands alone — you do not need to know why the site is broken to make it work again. You roll back the suspicious deploy, you flip off the suspect feature flag, you shift traffic away from the sick region, and the SLI recovers. Users are served. Only *then*, with the pressure off and the site healthy, do you sit down and calmly figure out what actually went wrong — and you'll diagnose far better, because nobody is bleeding while you think.

![Side by side comparison of a diagnose-first response that keeps users down for forty minutes versus a mitigate-first response that restores service in four minutes by rolling back before any root-cause work begins](/imgs/blogs/mitigate-first-diagnose-later-1.png)

By the end of this post you'll have a concrete mitigation toolkit — the fast, reversible levers that restore service without knowing the cause — a decision table for which lever to reach for, a feature-flag kill-switch snippet and a rollback runbook step you can adapt, a discipline for confirming the SLI actually recovered before you declare victory, and a clear map of the few dangerous cases where reflexive mitigation makes things *worse*. We'll ground all of it in the series' spine: you **detect** user pain, you **mitigate** to stop it, you **confirm** recovery, and only then do you **diagnose** and **engineer** the permanent fix. This post lives in the *responding* part of that loop, and it is where most teams either save or squander their reliability. If you haven't read the series opener, start with [the SRE mindset](/blog/software-development/site-reliability-engineering/reliability-is-a-feature-the-sre-mindset) — reliability is a number you engineer, and during an incident, mitigation is how you stop spending it.

## 1. Two different jobs that feel like one

The first thing to internalize is that an incident contains two completely different jobs that *feel* like a single job under stress.

**Mitigation** is the act of restoring service to users. Its success criterion is simple and measurable: the user-facing signal — the SLI, the thing that reflects whether customers are happy — returns to normal. Did errors drop back below threshold? Did latency come back down? Did the page clear? If yes, mitigation succeeded, regardless of whether anyone knows *why* it broke in the first place.

**Diagnosis** is the act of finding the cause. Its success criterion is understanding: you can explain the chain of events from trigger to symptom, you can point at the specific line of code or config or capacity wall, and you could prevent it from recurring. Diagnosis is what feeds the postmortem and the permanent fix.

These two jobs have different goals, different success criteria, different time pressures, and — crucially — they do not depend on each other in the direction people assume. You do *not* need diagnosis to perform mitigation. A rollback restores the previous known-good state whether or not you understand the bug in the new release. A feature-flag kill removes the broken feature whether or not you've found the null-pointer dereference inside it. The whole power of the mitigation toolkit is that every lever in it works *blind* — it targets the blast radius, not the root.

![A vertical stack showing detect then mitigate then confirm then diagnose then engineer as five separate stages, with mitigation and diagnosis drawn as distinct goals rather than one task](/imgs/blogs/mitigate-first-diagnose-later-5.png)

The reason these feel like one job is that in normal engineering work they *are* one job. When you're fixing a bug in a code review or a flaky test, you find the cause and then you fix it, and the fix *is* the resolution. There's no separate "make it work now" step because nobody is bleeding. The on-call brain imports that workflow into the incident, and it is exactly backwards: under fire, restore first, explain second.

Here's the cleanest way to hold it: **mitigation buys time; diagnosis spends it.** Mitigation is fast and cheap and reversible. Diagnosis is slow and expensive and open-ended. You do the cheap fast thing that stops the bleeding, then you spend as long as you need on the slow expensive thing — but you spend it with the site *up*. The order is the entire skill.

The "buys time" framing is worth taking literally, because it reframes what success looks like on the bridge. The goal of the first five minutes is not to *solve* the incident — it's to *change the clock you're racing*. Before mitigation, you're racing the user's clock: every second is downtime, error budget, and angry tweets. After mitigation, you're racing your *own* clock: the diagnosis can take an hour and nobody outside the room feels it. Mitigation converts an emergency into a problem. An emergency demands a fast, possibly-wrong decision under pressure; a problem allows a careful, correct decision in calm. The single most valuable thing a responder can do in the first minutes is to perform that conversion — to buy time — and the way you buy time is by pulling a reversible lever that restores the SLI. Everything downstream of that purchase is cheaper, because you bought it with the site up.

This also reframes the relationship between mitigation and the permanent fix. People sometimes worry that mitigating "lets the bug off the hook" — that if you roll back and the site is healthy, the pressure to fix the real problem evaporates and the bug lingers. The opposite is true on any team with a working postmortem process. Mitigation doesn't *resolve* the incident; it *de-escalates* it. The incident stays open, the bug stays tracked, the postmortem still happens — but all of that now occurs in daylight, with the contributing factors examined honestly rather than guessed at under fire. The danger of the bug lingering comes not from mitigating fast but from never writing the postmortem, which is a culture problem the mitigate-first discipline doesn't cause and doesn't excuse.

### Why "just find it real quick" is a trap

The most seductive sentence on any bridge is *"I think I almost have it — give me two minutes."* It is almost never two minutes. Live diagnosis under incident pressure has three properties that make it a trap.

First, **it's open-ended.** You don't know how long it'll take to find the cause, because if you knew where to look you'd already be looking there. "Two minutes" is a hope, not an estimate. Mitigation, by contrast, has a *bounded* cost: a rollback takes about as long as a deploy, a flag flip takes seconds, a traffic shift takes a minute or two. You can promise mitigation timelines. You cannot promise diagnosis timelines.

Second, **it accrues sunk cost.** Ten minutes into a live investigation, the team has invested effort and ego into a hypothesis. Abandoning it to "just roll back" feels like wasting that work. So people push on, and the ten minutes becomes thirty, all while users are down. The sunk-cost fallacy is brutal during incidents precisely because the clock is the user's, not yours.

Third, **stress degrades diagnosis.** You debug *worse* under pressure. Tunnel vision narrows your search, you stop reading the whole error and fixate on the first suspicious line, you skip the boring hypothesis (it's the deploy) for the interesting one (it's a race condition in the cache layer). Calm debugging with the site up is dramatically more effective than panic debugging with the site down. So even on pure diagnostic-quality grounds, mitigating first and diagnosing later gives you a *better* root cause, not just a faster recovery.

The rule that cuts through all of this is a stopwatch comparison: **if you can mitigate in 2 minutes and diagnosing will take 20, mitigate.** Make the comparison explicit out loud on the bridge. "We can roll back in four minutes. Does anyone have a confirmed fix in less than four? No? Then we roll back now and diagnose with the site up." That single sentence, spoken by the incident commander, ends more outages early than any amount of cleverness.

## 2. The mitigation toolkit: fast, reversible levers

A mitigation lever is a control you can pull that restores service *without knowing the cause*. The good ones share three properties: they're **fast** to apply (seconds to a few minutes), they're **reversible** (you can undo them if they don't help or make things worse), and they target the **blast radius** rather than the root — they remove or route around the broken thing instead of fixing it.

Here is the core toolkit, roughly in the order you should reach for them.

**Rollback.** The deploy is the single most common cause of incidents. Across the industry, the large majority of self-inflicted outages trace to a recent change — a code deploy, a config push, a dependency bump. So the highest-prior, highest-value lever is the rollback: if the incident started right after a deploy, revert to the previous known-good version *before* you diagnose. You don't need to know what's wrong with the new build. You only need to know it's new and the trouble started when it shipped. Roll back, watch the SLI, ask why later.

*When to reach for it:* the symptom started within minutes of a deploy and the symptom is broad (multiple endpoints, the whole service, a step-function jump in error rate). *How to do it well:* roll back to the *immediately previous* revision, not to "some version that worked last week" — the smaller the jump backward, the smaller the chance you reintroduce an old bug or skip past a needed migration. Watch the rollout actually finish (pods Ready, not just "rollback command accepted") before you read the SLI, because a half-completed rollback shows a confusing mixed signal. And keep the failed artifact around: roll back the *running* version, but don't delete or garbage-collect `v412`'s image, because you'll need it in staging an hour later to reproduce the bug calmly.

**Feature-flag kill switch.** If a single feature is implicated — a new recommendation widget, a new pricing rule, a new code path behind a flag — flipping the flag off turns the suspect feature off in seconds, without a deploy. This is the surgical version of a rollback: instead of reverting the whole release, you disable just the part that's likely on fire. It's faster than a rollback (no build, no redeploy) and more precise (the rest of the release keeps running).

*When to reach for it:* the symptom maps cleanly to one new feature, *and* that feature was shipped behind a flag (which is exactly why every risky feature should be). *How to do it well:* the kill switch only counts as a mitigation lever if flipping it propagates to every instance in seconds — a flag that the app reads only at process start is not a kill switch, it's a deploy in disguise. Wire the flag read to a short TTL or a push channel (covered in the runnable section below), and make the *off* state a tested fallback, not an untested code path you've never exercised in production. The most common failure of a "kill switch" is discovering mid-incident that the off branch was never actually tested and crashes differently than the on branch.

**Traffic shift / failover.** If the problem is localized to a region, an instance group, a canary version, or a single zone, route traffic away from it. Shift the load balancer to send users to the healthy region. Promote the old version's pool back to 100%. Fail over to the standby. The bad thing is still bad — you just stop sending users to it.

*When to reach for it:* the symptom is *localized* — one region is throwing errors while another is clean, the canary pool is sick while stable is fine, one availability zone shows the cliff and the others don't. The tell is asymmetry in your per-region or per-version SLI breakdown. *How to do it well:* before you shift, confirm the target can actually take the load — failing all traffic into a region that was only ever sized for half of it just moves the outage. A traffic shift is reversible and stateless when the service is stateless; it becomes dangerous the moment the thing you're failing over holds state (see the stateful-failover caveat in the "when not to" section). For a stateless web tier, shift freely; for a database, slow down.

**Drain.** A finer-grained traffic move: pull one sick instance out of the load balancer pool so it stops receiving requests. If a single node is wedged — a memory leak, a stuck thread pool, a corrupted local cache — draining it removes its bad responses from the user's experience while leaving the rest of the fleet serving normally.

*When to reach for it:* the per-instance breakdown shows one or two hosts misbehaving while the rest are healthy — one pod with a climbing memory graph, one node returning the bulk of the errors. *How to do it well:* drain rather than kill. Marking an instance unhealthy (cordon, remove from the endpoints list, set its readiness probe to fail) stops it serving users *without destroying its state*, so it's still there for diagnosis. This is the crucial difference from restart: a drained pod is a quarantined witness; a restarted pod is a destroyed crime scene. Drain when you suspect a single bad host but want to keep it for the autopsy.

**Scale out.** When the symptom is saturation — a load spike, a viral traffic event, a thundering herd — adding capacity absorbs the spike. More replicas means each one handles fewer requests, queues drain, latency falls. Scale-out doesn't fix *why* the load is high; it buys headroom so users stop seeing timeouts while you figure out the why.

*When to reach for it:* the symptom is latency-and-saturation, not errors-from-a-bug — p99 climbing, queue depth growing, CPU or connection pools pinned, request rate clearly up. The tell is that *nothing shipped* and the load graph is the thing that moved. *How to do it well:* scale aggressively past what the autoscaler would pick (the autoscaler reacts on a lag; during an incident you want headroom *now*, so jump to 3–4× and trim later), but first answer one question — *is the bottleneck the thing I'm scaling?* If checkout is slow because checkout is out of CPU, scale-out fixes it. If checkout is slow because the shared database behind it is saturated, more checkout replicas pour *more* load onto the same database and accelerate the collapse. Scale-out is the one fast lever that can actively backfire, and it backfires precisely when the bottleneck is downstream and shared.

**Restart — the blunt instrument.** Restarting the affected process or pod is the crudest lever. It often works (it clears a leaked-memory state, a deadlocked thread, a stuck connection pool) and it's fast. But it is the *least* informative: it masks the cause by wiping the very state you'd need to diagnose. Treat restart as a last resort among the fast levers, and when you do restart, **capture state first** (a heap dump, the current connection counts, the goroutine/thread dump) so you don't trade recovery for permanent blindness.

*When to reach for it:* the host or process is wedged in a way the other levers can't address — a deadlock, an exhausted local resource, a stuck event loop — and you've already ruled out the deploy and ruled out a downstream cause. *How to do it well:* never restart in a blind loop. A pod that crash-loops on a poison message or a bad config will crash again the instant it comes back, and each restart burns the evidence a little more. Capture a thread dump and a heap dump *before* the restart, snapshot the metrics and queue depths, and prefer draining-then-restarting one instance (so you can compare the restarted one against a still-wedged sibling you kept). Restart is motion; make sure it's also progress.

**Roll back the data / config.** Not every change is a code deploy. A bad config push — a wrong feature-flag default, a broken routing rule, a fat-fingered rate limit, a bad DNS or load-balancer config — causes outages just as readily as code. Revert the config to its last-good value. Config changes are often *faster* to revert than code (no build), which is exactly why config rollback belongs in the front of the toolkit alongside code rollback.

*When to reach for it:* the symptom started right after a config or routing change, or when there was no code deploy but *something* in the control plane moved — a limit, a rule, a DNS record, a load-balancer weight. Config pushes are easy to forget in the "did anything change?" question precisely because they don't feel like "a deploy," so your change-annotation stream must include them. *How to do it well:* the entire recovery time of a config-caused outage is usually dominated by one number — how fast can you revert the config to its last-known-good value — so the prerequisite is that config is versioned, that the last-good value is recorded, and that the revert is a single tested command. A config rollback you have to reconstruct by hand from memory is not a fast lever; it's an archaeology project at the worst possible time.

![A matrix listing six mitigation levers against what each one fixes, whether it is reversible, and how long it takes to apply, showing the fastest reversible levers apply in under two minutes](/imgs/blogs/mitigate-first-diagnose-later-2.png)

### The lever decision table

Reaching for the right lever fast is itself a skill, and you don't want to be inventing the decision tree at 14:04. Keep this table where the on-call can see it.

| Lever | What it fixes | Reversible? | Time to apply | Cost / caveat |
|---|---|---|---|---|
| **Rollback deploy** | A bad code release (the #1 cause) | Yes — roll forward later | 2–5 min | Loses any good changes in the same release |
| **Feature-flag kill** | One bad feature behind a flag | Yes — flip back on | < 60 sec | Only works if the feature was flagged |
| **Config rollback** | A bad config / routing / limit push | Yes — re-apply old config | < 60 sec–2 min | Must know the last-good value |
| **Traffic shift / failover** | A bad region, version, or zone | Yes — shift back | 1–3 min | Standby must be healthy and warm |
| **Drain instance** | One sick host in the pool | Yes — re-add to pool | < 60 sec | Doesn't help if all hosts are sick |
| **Scale out** | Saturation / load spike | Yes — scale back in | 2–10 min | Costs money; useless if the bottleneck is a shared dependency |
| **Restart** | Stuck process / leaked state | N/A — blunt | < 60 sec | Masks the cause; capture state first |

Read it as a lookup: *what's the symptom and what changed?* If something shipped and the symptom is broad, rollback or config rollback. If something shipped and the symptom maps to one feature, flag kill. If nothing shipped and the symptom is regional, failover. If nothing shipped and the symptom is saturation, scale out. If one host is misbehaving, drain. Restart only when nothing else fits and you've grabbed state first.

The two columns that drive the *order* you reach in are "Reversible?" and "Time to apply." The fastest reversible levers — flag kill, config rollback, drain — apply in under a minute and undo just as fast, so they sit at the front: if one of them fits the symptom, you try it first because the cost of being wrong is a handful of seconds. Rollback and traffic shift are a tier slower (a minute or several) but still cleanly reversible, so they're the second reach. Scale-out is reversible but costs money and can backfire when the bottleneck is shared, so it's situational. Restart is last because it's the only one whose "reversibility" is an illusion — you can restart the process again, but you cannot un-destroy the in-memory evidence you wiped. Sorting the table by *reversibility-then-speed* is exactly the priority order the discipline wants: cheap-and-undoable first, expensive-or-lossy last.

Notice the column that matters most for the discipline: **none of these levers requires a root cause.** Every "what it fixes" entry is a *blast-radius* description — "a bad region," "one sick host," "a load spike" — not a root-cause description. That's the whole point. You're treating the symptom's *location*, not its *mechanism*. A useful sanity check when you're about to pull a lever: can you state what it fixes without using the word "because"? "Roll back because the new query is slow" is a diagnosis dressed up as a mitigation — you don't know that yet. "Roll back because the symptom started at the deploy" is a blast-radius move that needs no theory of the bug. If your justification for a lever contains a causal claim you haven't proven, you've slipped back into diagnosing; pull the lever on the *correlation*, not the *explanation*.

## 3. Why blast-radius levers work without diagnosis

It's worth being precise about *why* these levers can possibly work when you don't know what's wrong, because the principle generalizes.

An incident has a structure: a **trigger** (the thing that changed or spiked) causes a **fault** (something is now broken internally) which produces a **symptom** (users experience errors or latency). Diagnosis is the work of tracing symptom → fault → trigger, all the way back. It's hard because the chain can be long and the arrows can be non-obvious — a deploy three services upstream causes a connection-pool exhaustion two services down that manifests as checkout timeouts.

Mitigation doesn't trace the chain. It *cuts* it. Every lever in the toolkit severs the path from trigger to symptom at some point:

- **Rollback / config rollback / flag kill** removes the *trigger*. If the bad thing isn't running anymore, the fault it caused stops being produced. You didn't fix the fault; you stopped feeding it.
- **Traffic shift / failover / drain** routes around the *fault*. The broken thing is still broken, but no user requests reach it, so the symptom disappears from the user's side.
- **Scale out** absorbs the *symptom* directly. If the fault is "we're overloaded," more capacity makes the overload smaller until it's below the pain threshold.

So the levers map cleanly onto the trigger → fault → symptom chain, and each works by attacking a different link without needing to understand the others. This is why you can mitigate blind: you're not solving the equation, you're disconnecting the circuit.

There's a formula hiding here worth making explicit. Define your time-to-mitigate as $\text{TTM}$ and your time-to-diagnose as $\text{TTD}$. User-facing downtime under "diagnose first" is roughly $\text{TTD} + \text{TTfix}$ — you eat the full diagnosis *and* the fix before users recover. Under "mitigate first," user-facing downtime is just $\text{TTM}$, because the diagnosis and the real fix happen with the site already up:

$$
\text{Downtime}_{\text{diagnose-first}} = \text{TTD} + \text{TT}_{\text{fix}}, \qquad \text{Downtime}_{\text{mitigate-first}} = \text{TTM}
$$

Since $\text{TTM}$ is small and bounded (a rollback) while $\text{TTD}$ is large and open-ended (a live investigation), the savings are enormous. The error budget you preserve is exactly the gap between these two — and the error budget is the currency of this whole discipline, as we cover in [the error budget post](/blog/software-development/site-reliability-engineering/the-error-budget-the-currency-of-reliability). Every minute of $\text{TTD}$ you avoid eating live is a minute of budget you keep.

#### Worked example: the budget you save by mitigating first

Take a service with a 99.9% availability SLO. That gives you a monthly error budget of about 43.2 minutes of downtime (0.1% of ~43,200 minutes in a 30-day month). Now an incident hits.

Under **diagnose-first**, the team spends 40 minutes finding the cause and 5 minutes shipping the fix: 45 minutes of user-facing downtime. That's *more than your entire month's budget gone in one incident* — you're now over budget, which (in a healthy SRE org) means a deploy freeze until you recover.

Under **mitigate-first**, the team rolls back in 4 minutes. User-facing downtime: 4 minutes. You've spent under 10% of the month's budget. The diagnosis and the real fix still happen — but they happen over the next hour with the site healthy, costing zero further budget. Same root cause found, same permanent fix shipped, but **4 minutes of downtime instead of 45.** That ~11× difference is the whole argument, in arithmetic.

## 4. "Is it the deploy?" — the first question

If there's one question to ask first on every incident, it's this: **did the symptom start right after a change?**

The reason this is the first question is statistical. The large majority of incidents are self-inflicted — caused by something *we* changed: a code deploy, a config push, a dependency upgrade, an infrastructure change, a flag flip. Random hardware failures and pure traffic surprises exist, but they're the minority. So the highest-information, fastest question you can ask is whether the incident correlates in time with a recent change. If it does, you have a one-step mitigation (revert the change) and a strong lead for diagnosis later.

This is the same logic as bisection in debugging: if you know the system was healthy at time T-and-before and sick at time T-and-after, the cause is overwhelmingly likely to be whatever happened *at* T. During an incident, the "commits" are your deploys and config pushes, and the "good/bad boundary" is the moment the SLI broke. Correlating the incident start with the deploy timeline *is* a coarse bisection — and if you want the fine-grained version for finding which commit in a release caused the bug after you've mitigated, see [binary-search your bug with bisection](/blog/software-development/debugging/binary-search-your-bug-with-bisection).

Make this correlation cheap to do. Your incident tooling should surface, on one screen: the SLI graph (when did it break?) overlaid with a **deploy / change marker stream** (what shipped and when?). If the error-rate cliff lines up with a deploy marker, you have your answer in seconds. Annotate deploys into Grafana as vertical lines and the incident commander can eyeball the correlation instantly:

```promql
# The user-facing SLI as a recording rule (good fraction over a 5m window).
# A sharp drop in this line is your "incident started here" marker.
record: sli:checkout_success:ratio_rate5m
expr: |
  sum(rate(http_requests_total{job="checkout",code!~"5.."}[5m]))
  /
  sum(rate(http_requests_total{job="checkout"}[5m]))
```

Overlay deploy events on the same Grafana panel as annotations (one annotation per deploy, tagged with the version), and the moment the SLI cliffs you can read off the version that shipped just before it. That visual correlation is the fastest diagnosis-adjacent signal you'll ever get, and it directly tells you which thing to roll back.

![A branching flow that starts from a Sev1 page, asks whether the incident started near a recent change, and routes to rollback or to traffic moves before checking whether the SLI recovered](/imgs/blogs/mitigate-first-diagnose-later-4.png)

The decision flow is short on purpose. Page fires → did it start near a change? → **yes**: roll back or kill the flag. **no**: shift traffic, drain, or scale out depending on the symptom shape. Either branch → did the SLI recover in 2–5 minutes? → if yes, you're mitigated and you diagnose calmly. If no, the change wasn't the (only) cause; widen the search and try the next lever. The whole flow is designed to reach a mitigation attempt within the first couple of minutes, not the first couple of *quarter*-hours.

### A note on "but what if the rollback is the wrong move?"

People resist rolling back because they're afraid of being wrong — what if the deploy *wasn't* the cause and you rolled back for nothing? Here's why that fear is misplaced: **the rollback is reversible.** If you roll back and the SLI doesn't recover, you've learned something valuable (it wasn't the deploy) at a cost of a few minutes, and you can roll forward again. The downside of a wrong rollback is small and recoverable. The downside of a *correct-but-too-late* diagnosis is large and unrecoverable (the downtime already happened). When one option is cheap-and-reversible and the other is expensive-and-irreversible, you take the cheap reversible one first. That's not just incident tactics; it's decision theory.

## 5. The reversibility principle

The deepest design principle inside the mitigation toolkit is **reversibility**: prefer mitigations you can undo over ones you can't.

A rollback you can roll forward, a flag you can flip back on, a traffic shift you can shift back, a scale-out you can scale back in — these are all *reversible*. If you pull the lever and it doesn't help (or makes things worse), you put it back and try something else. The cost of being wrong is bounded by the time to reverse, which is small.

Compare that to *irreversible* moves: deleting data to "clean up" a corrupted table, force-killing a process mid-write, truncating a queue, blowing away a cache that's expensive to rebuild, manually editing production state. If you do one of these and you're wrong — or even if you're right but it had a side effect you didn't anticipate — you cannot simply undo it. The cost of being wrong is unbounded and possibly permanent.

The principle: **under uncertainty, reach for reversible levers first.** During an incident, you are *always* under uncertainty — that's what an incident is. So the toolkit is deliberately stacked with reversible levers, and the irreversible moves are pushed to the very end, used only when (a) you actually understand the situation, and (b) the reversible levers have failed.

This is why "restart" sits awkwardly in the toolkit. A restart is technically reversible in the sense that you can start the process again, but it's *irreversible in the diagnostic sense*: it destroys the in-memory state that holds the evidence. You can't un-restart and get back the heap you needed to see. So treat restart with the caution you'd give any irreversible move: capture state first (thread dump, heap dump, metrics snapshot, the current values of any counters or queue depths), *then* restart. The recovery is reversible; the evidence loss is not.

A practical way to bake reversibility into your tooling is to make every mitigation lever come with its inverse pre-defined. Your rollback runbook should state the roll-forward command. Your flag-kill runbook should state how to re-enable. Your failover runbook should state the failback. If a lever doesn't have a documented, tested inverse, it's not a mitigation — it's a gamble, and you treat it accordingly.

There's a decision-theory backbone under this that's worth naming, because it tells you what to do under uncertainty in general, not just during incidents. When you must act with incomplete information, the right move is the one that **minimizes the cost of being wrong**, not the one that maximizes the chance of being right. A reversible lever has a small, bounded cost-of-being-wrong (the time to undo it), so even if it's the wrong guess, the regret is tiny. An irreversible move has an unbounded cost-of-being-wrong, so being wrong even once can dominate every right decision you ever made. During an incident you are *guaranteed* to be acting under uncertainty — you don't yet know the cause, that's what makes it an incident — so you are exactly in the regime where minimizing the cost of error beats maximizing the chance of correctness. This is why "reach for the reversible lever" is not timidity; it's the mathematically correct policy when you're forced to decide before you understand.

The same principle explains why reversibility should shape the systems you *build*, not just the levers you pull. A team that invests in one-command rollbacks, feature flags on every risky path, blue-green or canary deploys you can abort, and versioned config with a tested revert is a team that has manufactured reversibility into its operations. That investment pays off as cheaper incidents forever after, because it widens the set of mitigations that are fast-and-undoable. Conversely, a team whose only "rollback" is a manual hour-long redeploy has accidentally made its mitigations irreversible-in-practice (an hour is too long to undo under fire), which forces it back toward diagnose-first whether it wants to or not. The architectural choices that create reversibility — progressive delivery, flags, expand-and-contract migrations — are the precondition that makes the mitigate-first *discipline* even available. You can't reach for a reversible lever you never built.

## 6. A runnable mitigation toolkit

Principles are nice; let's make them copy-and-adapt real. Here are the artifacts for the four most-reached-for levers.

### Feature-flag kill switch

The fastest lever in the toolkit, because it needs no deploy. A kill switch is just a feature flag whose *off* state is a fast, safe fallback. The trick is to wire it so that flipping it takes seconds and the application picks it up immediately (poll or push), not on the next deploy.

```python
# A minimal kill-switch pattern. The flag is read from a fast store
# (a config service / feature-flag service) on every request, cached
# briefly so a flip propagates within seconds, not minutes.
import time

class KillSwitch:
    def __init__(self, flag_client, name, ttl_seconds=5):
        self._client = flag_client
        self._name = name
        self._ttl = ttl_seconds
        self._cached = None
        self._fetched_at = 0.0

    def is_killed(self) -> bool:
        now = time.monotonic()
        if self._cached is None or now - self._fetched_at > self._ttl:
            # Fail SAFE: if the flag store is unreachable, default to the
            # safe path. For a kill switch, "killed" is usually the safe path.
            try:
                self._cached = self._client.get_bool(self._name, default=False)
            except Exception:
                self._cached = self._cached if self._cached is not None else False
            self._fetched_at = now
        return self._cached

# Usage at the call site of the risky feature:
recs_kill = KillSwitch(flag_client, "checkout.recommendations.kill")

def render_checkout(cart, user):
    page = base_checkout(cart, user)
    if not recs_kill.is_killed():
        # The suspect feature. If it's on fire, we flip the kill switch and
        # this whole branch stops running within ~5 seconds, no deploy.
        page.recommendations = compute_recommendations(user)
    return page
```

To mitigate: flip `checkout.recommendations.kill` to `true` in your flag UI or CLI. Within the TTL (5 seconds here), every instance stops calling `compute_recommendations`, the suspect path is gone, and the page still renders. To roll forward later, flip it back. Note the two design decisions that make this a *good* kill switch: a short TTL so flips propagate fast, and a *fail-safe* default so that if the flag store itself is unreachable during the incident, you don't accidentally re-enable the broken feature.

### Rollback runbook step

A rollback is only a fast mitigation if it's *rehearsed and scripted*. The worst time to figure out your rollback command is mid-incident. Put the exact step in the runbook:

```bash
# RUNBOOK: Roll back checkout-service to the previous known-good release.
# Precondition: incident correlates with a recent deploy (check the deploy
# annotations on the SLI dashboard). Rollback is REVERSIBLE — note the
# roll-forward command at the bottom.

# 1. Identify the current and previous revision.
kubectl -n prod rollout history deployment/checkout-service

# 2. Roll back to the immediately previous revision.
kubectl -n prod rollout undo deployment/checkout-service

# 3. Watch the rollout complete (don't declare victory until pods are Ready).
kubectl -n prod rollout status deployment/checkout-service --timeout=120s

# 4. CONFIRM RECOVERY on the SLI (see the "did the SLI recover?" check below).
#    Do NOT close the incident until the user-facing signal is back to baseline.

# ROLL FORWARD (if the rollback did NOT help — i.e. it wasn't the deploy):
#   kubectl -n prod rollout undo deployment/checkout-service --to-revision=<N>
#   ...then move to the next lever (traffic shift / scale out).
```

The runbook encodes the discipline: roll back, *watch the rollout actually finish*, then *confirm on the SLI*, and — critically — it documents the roll-forward so the responder knows the move is reversible and isn't afraid to pull it.

### Traffic shift / failover

If the symptom is regional or version-scoped, route away from the bad thing. With a service mesh or weighted load balancer, this is a weight change — itself a config change you can revert:

```yaml
# Argo Rollouts: abort an in-progress canary and send 100% back to stable.
# This is the "shift away from the bad version" lever. It's reversible:
# you resume or re-promote once you've diagnosed.
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: checkout-service
spec:
  strategy:
    canary:
      # During an incident on the canary, abort sends all traffic to stable.
      # CLI: kubectl argo rollouts abort checkout-service -n prod
      steps:
        - setWeight: 10
        - pause: { duration: 5m }
        - setWeight: 50
        - pause: { duration: 5m }
        - setWeight: 100
```

```bash
# Mitigation: abort the canary — all live traffic snaps back to the
# last stable version. Reversible: 'promote' resumes once it's safe.
kubectl argo rollouts abort checkout-service -n prod
# Roll forward later:
# kubectl argo rollouts promote checkout-service -n prod
```

For a region-level failover, the lever is a DNS or global-load-balancer weight change that drains the bad region. The progressive-delivery machinery that makes both canaries and fast rollbacks possible is covered in the planned sibling post on deploying safely with progressive delivery (`deploying-safely-progressive-delivery`); if you're shipping without a one-command rollback and a canary you can abort, building that is the highest-leverage reliability investment you can make, because it's what turns "mitigate first" from a slogan into a four-minute reality.

### Scale out to absorb a spike

When the symptom is saturation, add capacity. With an autoscaler this may be automatic, but during an incident you often want to *manually* scale beyond what the autoscaler would choose, fast:

```bash
# Mitigation for a load spike: scale out NOW, faster than the HPA would.
# Reversible: scale back in once the spike passes / root cause is fixed.
kubectl -n prod scale deployment/checkout-service --replicas=40
# (was 12) — watch queue depth and p99 fall as capacity absorbs the load.

# Roll back in after recovery:
# kubectl -n prod scale deployment/checkout-service --replicas=12
```

Scale-out is reversible and fast, but note the caveat from the table: it only helps if the bottleneck is the thing you're scaling. If checkout is slow because a *shared* database is saturated, adding more checkout replicas makes it *worse* — more clients hammering the same overloaded dependency. That's the connection to [cascading failures and bulkheads](/blog/software-development/system-design/cascading-failures-circuit-breakers-and-bulkheads): when the bottleneck is downstream, the right lever is load-shedding (reject some traffic at the edge to protect the dependency), not blind scale-out. Knowing *which* lever fits the saturation shape is exactly the judgment the decision table is meant to encode.

## 7. Confirm the mitigation actually worked

Pulling the lever is not the same as fixing the problem. The discipline that separates a mitigated incident from a *prematurely-closed* one is **confirming the SLI recovered before declaring victory.**

The failure mode is everywhere: the team rolls back, sees the deploy revert succeed in the CI tool, and announces "mitigated!" — but never actually checks the user-facing signal. Sometimes the rollback didn't help (it wasn't the deploy). Sometimes it helped partially (errors dropped from 8% to 2% — better, but still paging). Sometimes it helped and then the symptom came back (the underlying load is still climbing). In every case, declaring victory off the *action* rather than the *result* leads to a second page minutes later, now with a confused team that thought it was done.

So the confirmation step is non-negotiable: **watch the SLI return to baseline and hold there for a couple of windows before you stand down.** Not the action's success indicator — the *user's* signal. Did the error ratio drop below threshold? Did p99 latency come back to normal? Did the burn rate fall back under 1×? And did it *stay* there?

![A before and after comparison of the user facing signals during an incident versus after mitigation holds, showing error ratio, p99 latency, and burn rate all returning to baseline](/imgs/blogs/mitigate-first-diagnose-later-6.png)

Make the check concrete with the same SLI you alerted on. If you paged on a multi-window burn-rate alert, you confirm recovery on the *same* burn rate falling back under 1×:

```promql
# Did the SLI recover? Watch the burn rate fall back under 1x.
# burn_rate = observed_error_ratio / (1 - SLO)
# For a 99.9% SLO, the budget consumption rate is normal when this < 1.
(
  1 - sli:checkout_success:ratio_rate5m
)
/ (1 - 0.999)
```

Read it live: during the incident this expression was, say, 40 (you were burning budget 40× faster than sustainable). After a successful mitigation it should fall under 1 and *stay* under 1. If it drops to 8 and plateaus, you're better but not recovered — the lever helped partially, keep going. If it drops under 1 and then climbs back up, the mitigation didn't address the real driver — the symptom is returning, escalate.

A simple bash check you can drop in a runbook, querying Prometheus directly, so the on-call has an objective "are we recovered?" answer instead of a vibe:

```bash
# Query the live burn rate; recovery means it's comfortably under 1.
PROM=http://prometheus.prod:9090
BURN=$(curl -s "$PROM/api/v1/query" \
  --data-urlencode 'query=(1 - sli:checkout_success:ratio_rate5m) / (1 - 0.999)' \
  | jq -r '.data.result[0].value[1]')

echo "Current burn rate: $BURN"
awk -v b="$BURN" 'BEGIN { exit !(b < 1.0) }' \
  && echo "RECOVERED: burn rate < 1x. Hold one more window, then stand down." \
  || echo "NOT recovered: burn rate still >= 1x. Stay engaged, try next lever."
```

The rule for closing: **two consecutive healthy windows.** One window dipping healthy can be noise; two in a row holding under threshold means the mitigation took. Only then do you transition the bridge from "mitigating" to "diagnosing with the site up." This is also the moment you write the first real timeline entry: *14:08 — rolled back to v411, burn rate under 1× and holding, service restored.* That timestamp is your MTTM (mean time to mitigate), and it's the number that actually matters to users — far more than MTTR-to-full-resolution, which includes the calm diagnosis nobody felt.

## 8. Worked example: a Sev1 from a bad deploy

Let's walk the whole discipline end to end on a realistic Sev1, with the clock running.

#### Worked example: rolled back in 4 minutes, diagnosed calmly in the next hour

**14:02** — Release `v412` of `checkout-service` rolls out to production. Routine deploy, passed CI, passed the canary's first weight step.

**14:04** — The multi-window burn-rate alert fires Sev1. The 1-hour window shows a 40× burn rate; the 5-minute fast window is hot too. Error ratio on checkout is 8%, p99 latency is 4.2s (normally 240ms). Three engineers join the bridge; an incident commander is assigned.

**14:05** — The IC asks the first question: *"Did this start right after a change?"* The on-call pulls the SLI dashboard. The error-rate cliff lines up *exactly* with the `v412` deploy annotation at 14:02. That's the answer. Nobody knows *why* `v412` is broken — and it does not matter yet.

**14:06** — The IC makes the call out loud: *"We can roll back in about three minutes. Does anyone have a confirmed fix faster than that? No? Then we roll back now and diagnose with the site up."* The on-call runs the rollback runbook step: `kubectl rollout undo deployment/checkout-service`.

**14:08** — The rollout to `v411` completes, pods Ready. The on-call runs the burn-rate confirmation query. Burn rate has fallen from 40× to under 1× and is holding. Error ratio is back to 0.05%, p99 back to 240ms. The IC marks the incident **mitigated at 14:08 — MTTM 4 minutes.** The bridge exhales. Users are checking out again.

**14:10 – 15:10** — *Now* the team diagnoses, calmly, with the site healthy. They pull `v412` into a staging environment, reproduce the failure, and find it: a new query in the recommendations path was missing an index hint and table-scanning under production cardinality, exhausting the database connection pool and starving the entire checkout path. They write a fixed `v413`, get it reviewed properly (no panic), and ship it through the canary that evening.

Now compare the counterfactual. Had the team **diagnosed first**, they'd have spent that 14:10–15:10 hour of investigation *with the site down* — call it 40 minutes to find the connection-pool exhaustion under stress (slower than the calm 60 because panic degrades debugging) plus a few minutes to fix-forward. Roughly **40+ minutes of user-facing downtime instead of 4.** Same bug, same engineers, same fix. The only difference was the *order*: stop the bleeding, then find the wound.

![A left to right timeline showing a deploy at 14:02, a Sev1 page at 14:04, correlation to the deploy at 14:05, a rollback at 14:06, confirmed recovery at 14:08, and calm diagnosis starting at 15:10](/imgs/blogs/mitigate-first-diagnose-later-3.png)

The measured proof, stated honestly: **MTTM 4 minutes vs ~40 minutes** of downtime a live diagnosis would have cost — a ~10× reduction in user-facing impact for this incident, and roughly 10% of the monthly error budget spent instead of the whole thing plus an overage. These are illustrative numbers for one realistic incident, but the *ratio* — minutes to mitigate versus tens of minutes to diagnose — is consistent with what every mature on-call team observes once they adopt the discipline.

## 9. Worked example: a load spike absorbed by scale-out

Not every incident is a deploy. Here's the second canonical shape — saturation — and the levers shift accordingly.

#### Worked example: scale out and shed load, find the viral traffic after

**20:31** — A creator with millions of followers posts a link to a product page. Traffic to the product and checkout services begins climbing steeply.

**20:34** — The burn-rate alert fires. p99 latency is climbing past 3s, error ratio is at 4% and rising, the request queue depth on checkout is growing. Crucially: **no deploy happened.** The "did it start near a change?" question comes back *no*. The SLI cliff doesn't line up with any deploy marker — it lines up with a traffic spike (request rate is up 6× in three minutes).

**20:35** — Wrong instinct on the bridge: *"Let me find which endpoint is slow."* The IC redirects: *"It's a load spike, not a bug. We don't need to know which query is slowest. Scale out and shed load now; we'll profile the hot path after."*

**20:36** — Two reversible levers in parallel. First, **scale out**: `kubectl scale deployment/checkout-service --replicas=40` (from 12). At 12 replicas the fleet was sized for roughly 6,000 requests/second of headroom and the spike had pushed offered load to about 21,000 requests/second — a ~3.5× overload, which is exactly why the queues were filling and p99 was past 3s. Going to 40 replicas restores headroom to roughly 20,000 requests/second, enough to serve the spike with margin. Second, **shed load** at the edge: enable the rate limiter / load-shedder so excess requests get a fast, polite 429 with a retry hint rather than piling into a queue that makes *everyone* slow. Shedding the top ~10% of traffic during the gap before the new replicas come Ready keeps the served 90% fast instead of letting 100% degrade together. Shedding a fraction of traffic to keep the rest fast is the saturation-incident version of protecting the blast radius: a fast 429 to one in ten users is vastly better than a 4-second timeout for all ten.

**20:40** — Capacity catches up. Queue depth falls, p99 drops back under 300ms, error ratio under threshold for served traffic. Burn rate back under 1× and holding. **Mitigated.** The viral traffic is still coming — but the system is now sized to absorb it and the shedder is protecting the tail.

**21:00 onward** — Calm diagnosis: the team confirms the root "cause" is simply organic viral demand (not a bug), reviews whether the autoscaler's reaction was too slow (it was — they tune the HPA's scale-up policy and target a lower CPU threshold so it reacts to spikes sooner), and files a follow-up to pre-warm capacity when marketing knows a big push is coming.

The measured angle: a saturation incident absorbed with the SLI restored in ~6 minutes via reversible levers (scale-out + shed-load), versus the alternative of chasing "which endpoint is slow" for twenty minutes while the queue grew and *every* user suffered. The principle held even though not a single line of code was the cause: **mitigate the symptom's blast radius first (absorb and shed the load), diagnose the driver after (it was viral demand; the autoscaler was too slow).** Note that scale-out only worked because the bottleneck was checkout's own capacity; if the database had been the wall, scale-out would have backfired and load-shedding alone would have been the lever — which is the judgment the next section is about.

## 10. When NOT to mitigate blindly

The discipline is "mitigate first," not "mitigate *recklessly*." There are a handful of failure modes where the reflexive fast mitigation makes things *worse*, and a senior responder knows them cold so the reflex doesn't fire in exactly the wrong place.

![A matrix of four dangerous cases where a reflexive mitigation backfires, listing the reflex move, why it backfires, and the safer action for data corruption, schema migrations, crash loops, and stateful failover](/imgs/blogs/mitigate-first-diagnose-later-8.png)

**Data corruption.** If the incident is *writing bad data* — a deploy that's corrupting rows, double-charging customers, or mangling records — then rolling back the *code* does not undo the bad data already written, and worse, the few minutes you spend rolling back are minutes of *continued corruption*. Here the first mitigation is often to **stop the writes** — put the service in read-only mode, disable the writing path, or pause the queue consumer — *even before* you roll back. Stopping the bleeding here means stopping the *corruption*, not restoring write availability. A brief read-only outage is vastly cheaper than an hour of corrupted writes you then have to reconcile. Know this case: when bad data is the symptom, "available" is not the goal; "not making it worse" is.

The reason data corruption inverts the usual priorities is that availability is recoverable and data integrity often is not. If you go read-only for ten minutes, users are annoyed and then they're fine — the moment you re-enable writes, availability is fully restored, no residue. But if you stay *available* while corrupting for ten minutes, you've created ten minutes of bad rows that now have to be found, quarantined, and reconciled — and every downstream system that read those rows (caches, search indexes, analytics, a partner's webhook) may have propagated the corruption further. The cleanup can take days and may never be perfect. So the cost asymmetry is enormous and runs the opposite way from a normal outage: here, *more uptime makes the incident worse*. The reflex to "restore service fast" is dangerous precisely because restoring the *write* path is restoring the thing that's doing damage. The senior move is to recognize the symptom shape — "are we producing wrong data, or just failing requests?" — within the first minute, because that single distinction flips the entire playbook. Failing requests: mitigate toward availability. Corrupting data: mitigate *away* from availability, freeze the writes, and only then untangle what happened.

**Schema migrations / forward-only changes.** If a release included a database migration that's already applied, rolling back the *code* can leave the app talking to a schema it no longer matches — the rollback can be *more* broken than the bug. This is why the safe pattern is expand-and-contract (backward-compatible migrations) so that a code rollback is always safe; if you don't have that, a code rollback against a migrated schema is an irreversible-ish trap, and forward-fixing (ship a corrected version) may be the only safe move. Know whether your last release touched the schema before you reflexively roll back.

**Crash loops you don't understand.** Restarting a crash-looping pod over and over (or worse, auto-remediating it) can mask a cause that will only get worse — an instance that crashes on a poison message will crash again the moment it's restarted, and you've burned budget and evidence for nothing. Don't auto-remediate a crash loop you don't understand. **Drain** the bad instance (pull it from the pool so it stops serving, but don't kill it — keep it for diagnosis), then investigate why it loops. Restarting in a loop is motion, not progress.

**Stateful failover with replica lag.** Failing over a database or stateful service to a standby is a great mitigation *if* the standby is current. If the replica is lagging, a blind failover can lose the writes between the last-replicated point and the failure — you've traded an outage for *data loss*, which is usually worse. Check the replication lag (your RPO — recovery point objective) before you fail over a stateful system. For the mechanics of replication lag and failover safety, the database series covers it directly; the SRE point is just: **stateful failover is not a free reversible lever the way a stateless traffic shift is.**

The unifying rule: **a fast mitigation is safe when it's reversible and stateless; it's dangerous when it's irreversible or touches state.** Code rollbacks, flag kills, traffic shifts, drains, and scale-out are stateless and reversible — reach for them freely. Anything that touches *data* — data rollback, stateful failover, "cleaning up" corrupted records, truncating queues — is in the dangerous category, and there you slow down just enough to confirm you're not making it worse. The two-minute pause to ask "does this touch state?" is the one exception to "don't pause" that's always worth it.

## 11. War story: the deploy-correlation discipline at scale

The "mitigate first, diagnose later" discipline is not a fringe opinion — it's the explicit, documented practice of the teams that operate the largest systems in the world.

Google's SRE practice codifies it directly. The *Site Reliability Engineering* book's chapters on managing incidents and emergency response are blunt about it: **the first priority during an incident is to stop the bleeding, restore service, and preserve evidence for the root-cause analysis — in that order.** Restoring service comes *before* root-causing. Their guidance to roll back a suspect change rather than debug it live is exactly the principle in this post: a rollback is a fast, reversible mitigation that doesn't require understanding the bug. The diagnosis happens afterward, in the postmortem, with the pressure off. This is also why Google's culture invests so heavily in fast, safe rollbacks and progressive delivery — the *ability* to mitigate in minutes is what makes "mitigate first" possible. If your rollback takes an hour, "mitigate first" isn't available to you, and fixing *that* is the prerequisite reliability investment.

The same discipline shows up, painfully, in the famous outages where teams *didn't* follow it. Consider the general shape of a thundering-herd / retry-storm outage (a pattern documented across many large providers): a brief blip causes clients to retry, the retries amplify load, the amplified load makes the blip worse, and the system spirals. Teams that *diagnosed first* — trying to understand the precise feedback loop while it raged — often stayed down for hours. The mitigation that actually worked was almost always a fast, blunt, blast-radius lever: shed load aggressively (reject most traffic so the system can recover), or take the whole thing down and bring it back up cold without the retry storm. You don't need to fully understand the amplification dynamics to *shed the load*; you just need to stop the herd. The understanding — and the permanent fix (backoff + jitter, retry budgets, circuit breakers) — comes after, in the postmortem. The retry-amplification mechanics and the breakers that prevent them are covered in [cascading failures, circuit breakers, and bulkheads](/blog/software-development/system-design/cascading-failures-circuit-breakers-and-bulkheads).

The retry storm is a particularly instructive case because it's where diagnose-first is *most* tempting and *most* fatal. It's tempting because the system looks busy and complicated — dashboards everywhere are red, every service is implicated, and the urge to "understand the feedback loop" is overwhelming. It's fatal because the loop is *self-sustaining*: while you study it, it keeps amplifying, and the load you'd need to reason about keeps changing under you. The cruel twist is that a retry storm often has *no single root cause to find* in the usual sense — the triggering blip may have been trivial and already gone, and what's keeping you down is the herd itself. So a team that insists on finding "the cause" can be searching for something that no longer exists while the actual problem (the amplification) rages on untreated. The only thing that breaks the loop is to break the loop: shed enough load that the system gets headroom to drain its queues and recover, or cycle the whole thing cold so the synchronized retries scatter. This is the clearest possible demonstration that mitigation and diagnosis are different jobs — here, you can fully mitigate (shed load, recover) with the root cause permanently unknown, because the root cause stopped mattering the moment the storm became self-sustaining.

And the most common real outage of all — the **bad config push** — is the purest illustration. Many of the largest internet outages on record trace to a single configuration change rolled out globally: a bad routing config, a malformed rule, a wrong limit, a bad DNS entry. In the postmortems, the recovery time is dominated almost entirely by one number: **how fast could they revert the config?** When the config rollback was fast and tested, recovery was minutes. When it wasn't — when reverting required rebuilding state, or the config system itself was implicated, or nobody had practiced the rollback — recovery stretched to hours. The lesson is the series' lesson: the *capability* to mitigate fast (a one-command, tested config rollback) is the reliability investment; the *discipline* to use it before diagnosing is the on-call skill. (These are accurate as general patterns drawn from public postmortems; I'm describing the shape, not quoting a specific company's exact minute counts.)

## 12. How to reach for this (and when not to)

Mitigate-first is a near-universal default during incidents, but like every practice it has edges. Here's the decisive guidance.

**Reach for mitigate-first when:**
- Users are actively in pain (the SLI is broken) and you have a fast, reversible lever available. This is the overwhelming majority of incidents.
- The incident correlates with a recent change — roll it back, full stop, and diagnose later.
- You're tempted to "just find it real quick." That temptation is the *signal* to mitigate instead.
- Two incidents overlap, or the on-call is alone and stretched. Mitigation is cheap cognitive load (pull a known lever); live diagnosis is expensive. When you're overloaded, mitigate to buy yourself room.

![A decision tree that asks whether the incident followed a recent change or is load driven, routing to rollback or flag kill on one branch and scale out or drain on the other](/imgs/blogs/mitigate-first-diagnose-later-7.png)

**Slow down (don't mitigate blindly) when:**
- The mitigation touches **state** — data rollback, stateful failover, deleting or truncating anything. Take the two minutes to confirm you won't lose or corrupt data. The "when not to" matrix above is your checklist.
- The symptom is **data corruption** — stop the writes first; restoring availability while still corrupting is the wrong goal.
- You're about to do something **irreversible**. If the lever has no documented, tested inverse, treat it as a last resort, not a first reach.

**Don't mitigate at all (this isn't your incident to mitigate) when:**
- The "incident" is a **false page** — the SLI is fine and the alert is broken. Don't roll back a healthy service because a flaky alert fired. Confirm there's real user pain (look at the actual SLI, not just the alert) before pulling any lever. Fixing the noisy alert is a separate job — see [alerting that doesn't cry wolf](/blog/software-development/site-reliability-engineering/alerting-that-doesnt-cry-wolf).
- The degradation is **expected and within budget** — a planned maintenance window, a known graceful degradation that's working as designed. Mitigating a system that's behaving correctly just adds churn.

The cost of mitigate-first, stated honestly: you sometimes mitigate the *wrong* thing (roll back a deploy that wasn't the cause) and lose a few minutes. That cost is real but small and reversible — which is exactly why the discipline is safe to default to. The cost of *not* mitigating first — the open-ended live diagnosis with users down — is large and irreversible. When the downside of acting is small-and-reversible and the downside of waiting is large-and-permanent, you act.

A final note on culture: this discipline only works if the team is **not punished for rolling back.** If reverting a deploy is treated as an admission of failure, engineers will resist it and diagnose-first to "prove" the deploy was fine. A healthy on-call culture treats a fast rollback as a *win* — service was restored in minutes — and the diagnosis-and-fix as the calm follow-up it should be. This connects directly to blameless postmortems: the same culture that makes people comfortable rolling back fast is the culture that makes them comfortable telling the truth afterward.

## 13. Wiring it into your incident process

Mitigate-first isn't just an individual reflex; it's a process you can build into how incidents run. Three concrete hooks.

**Make "is it the deploy?" the first scripted question.** Your incident bridge template should literally have, as step one after assigning an IC: *"Did the SLI break right after a change? Check the deploy annotations."* Scripting it means the panicked brain doesn't have to remember it. The anatomy of how an incident is structured and run — roles, bridge, timeline — is the subject of the sibling posts on the anatomy of an incident (`the-anatomy-of-an-incident`) and incident command (`incident-command-staying-calm-under-fire`); the mitigate-first reflex lives inside that structure as the IC's first decision after triage.

**Pre-stage the levers.** Every service should have, in its runbook, the exact command for each applicable lever: the rollback command, the flag names and how to kill them, the failover/abort command, the scale-out command, each with its documented inverse. A lever you have to *invent* during an incident is not a fast mitigation. The whole toolkit should be copy-pasteable from the runbook in seconds.

**Measure MTTM separately from MTTR.** Track time-to-mitigate (when did user pain stop?) as a distinct metric from time-to-resolve (when was the permanent fix shipped?). MTTM is the number users feel; MTTR-to-full-resolution includes the calm diagnosis nobody noticed. A team adopting mitigate-first sees MTTM drop sharply (because they stop diagnosing live) while MTTR-to-resolution may stay the same or even rise (because the diagnosis now happens carefully). That's the *correct* shape: short user pain, careful permanent fix. If you only measure one number, measure MTTM — it's the one that maps to the error budget.

#### Worked example: what the metrics look like before and after

Take a team that, before adopting the discipline, diagnosed-first. Their incident records over a quarter show a median user-facing downtime of ~38 minutes per Sev1/Sev2, because the median incident involved 30-ish minutes of live root-causing before anyone acted.

After adopting mitigate-first — scripting the deploy question, pre-staging levers, treating rollback as a win — their records show a median **MTTM of ~6 minutes**: most incidents are now a fast rollback or flag kill, confirmed on the SLI, with diagnosis moved to the calm hour afterward. User-facing downtime per incident dropped roughly **6×**. Their MTTR-to-full-resolution barely moved (the diagnosis still takes about as long — it's just no longer on the user's clock), and the quality of their postmortems actually *improved*, because people were diagnosing calmly instead of under fire. Error-budget consumption per incident fell proportionally, which let them ship faster the rest of the quarter without freezing on overage. These are illustrative figures for a representative team, but the *direction and magnitude* — MTTM down several-fold, MTTR-to-resolution flat, postmortem quality up — is the consistent signature of teams that make the switch.

## 14. Stress-testing the discipline

A practice is only trustworthy if it survives the hard cases. Let's stress-test mitigate-first against the scenarios that break naive versions of it.

**What if the rollback doesn't help?** Then it wasn't (only) the deploy — and you've learned that in ~4 minutes at the cost of a reversible action. Roll forward if needed, move to the next lever (traffic shift, scale out), and re-run the confirmation check. The reversibility is what makes a wrong guess cheap. You haven't lost the diagnosis option; you've just ruled out one hypothesis fast.

**What if two incidents overlap?** Mitigate-first scales *better* than diagnose-first under overlap, because pulling a known lever is low cognitive load while live diagnosis is high. With two fires, you can't afford to deep-dive either — you mitigate both with fast levers (roll back both suspect changes, shed load), stabilize, and *then* untangle which incident is which. Diagnose-first collapses under overlap; mitigate-first is what keeps you afloat.

**What if the on-call is asleep / it's 3am / they're junior?** This is mitigate-first's strongest case. A scripted lever (roll back the thing that just deployed) is something a tired or junior engineer can execute correctly from the runbook. A live diagnosis at 3am is where tired engineers make things worse. The discipline is *most* valuable exactly when the responder is least equipped to diagnose — it lets them restore service by following the runbook, then escalate the diagnosis to someone fresh.

**What if the error budget is already spent?** Then you're under a deploy freeze and the *only* changes going out are reliability fixes — which means mitigate-first matters *more*, because every minute of downtime is now coming out of an already-negative budget. You mitigate ruthlessly to stop further burn, and you do *not* improvise risky live fixes that might dig the hole deeper.

**What if the dependency is down for two hours?** A third-party or downstream dependency outage isn't something you can roll back — you didn't cause it. But the mitigation toolkit still applies: fail over to a fallback or cached path, shed the load that depends on the broken dependency, degrade gracefully (serve a reduced experience that doesn't need the dependency). You mitigate the *user impact* of the dependency outage without being able to fix the dependency itself. The blast-radius principle holds: route around the broken thing.

**What if mitigating would lose data?** Then you're in the "when not to" territory — slow down, confirm the RPO, and prefer the mitigation that preserves data (stop writes, read-only mode) over the one that restores availability at the cost of data. This is the one place where "stop the bleeding" means stopping the *data* bleeding, not the availability bleeding.

In every hard case, the discipline holds — sometimes by reaching for a different lever, occasionally by deliberately *not* pulling a state-touching lever. What never holds up is the instinct to root-cause live while users are down. That instinct loses every stress test.

## Key takeaways

- **Mitigation and diagnosis are two different jobs.** Mitigation restores service; diagnosis finds the cause. They have different goals and don't depend on each other in the direction people assume — you can mitigate blind.
- **Mitigate first, diagnose later.** Every minute spent root-causing live is a minute users are down. Stop the bleeding, then find the wound — calmly, with the site up.
- **The deploy is the #1 cause.** Ask "did it start right after a change?" first. If yes, roll back before you diagnose. Correlating the incident start with the deploy timeline is your fastest coarse bisection.
- **The toolkit is reversible, blast-radius levers:** rollback, feature-flag kill, traffic shift / failover, drain, scale out, config rollback, and restart (the blunt last resort — capture state first). None requires a root cause.
- **The 2-vs-20 rule:** if you can mitigate in 2 minutes and diagnosing will take 20, mitigate. Say the comparison out loud on the bridge.
- **Reversibility is the deepest principle.** Prefer levers you can undo (a rollback you can roll forward) over irreversible ones. Under incident uncertainty, reach for reversible first.
- **Confirm on the SLI, not the action.** Don't declare victory because the rollback "succeeded" — watch the user-facing signal return to baseline and hold for two windows.
- **Know the dangerous cases.** Data corruption (stop writes first), schema migrations (forward-fix), crash loops (drain, don't restart-loop), and stateful failover with replica lag (check RPO) are where blind mitigation backfires. Stateless+reversible = reach freely; touches state = slow down.
- **Measure MTTM separately.** Time-to-mitigate is the number users feel and the one that maps to the error budget. Expect it to drop several-fold while MTTR-to-resolution stays flat — that's the correct shape.

## Further reading

- [Reliability is a Feature: The SRE Mindset](/blog/software-development/site-reliability-engineering/reliability-is-a-feature-the-sre-mindset) — the series opener and the define→measure→budget→respond→learn loop this post lives inside.
- [The Error Budget: The Currency of Reliability](/blog/software-development/site-reliability-engineering/the-error-budget-the-currency-of-reliability) — why every minute of downtime you avoid by mitigating fast is budget preserved.
- [Alerting That Doesn't Cry Wolf](/blog/software-development/site-reliability-engineering/alerting-that-doesnt-cry-wolf) — confirm there's real user pain (and that the page isn't a false alarm) before you pull any lever.
- [Binary-Search Your Bug with Bisection](/blog/software-development/debugging/binary-search-your-bug-with-bisection) — the fine-grained version of "is it the deploy?" for finding the exact bad commit, after you've mitigated.
- [Cascading Failures, Circuit Breakers, and Bulkheads](/blog/software-development/system-design/cascading-failures-circuit-breakers-and-bulkheads) — the architecture-level resilience (load shedding, breakers, bulkheads) that makes some mitigations possible and prevents retry storms.
- Sibling posts (planned in this series): *The Anatomy of an Incident*, *Incident Command: Staying Calm Under Fire*, and *Deploying Safely with Progressive Delivery* — the incident structure the mitigate-first reflex lives inside, and the fast-rollback machinery that makes a four-minute mitigation possible.
- Google's *Site Reliability Engineering* book — the chapters on Managing Incidents and Emergency Response, which codify "stop the bleeding and restore service before root-causing."
- The Prometheus and Alertmanager documentation, and the SRE Workbook's chapter on multi-window burn-rate alerting — for the SLI recording rules and burn-rate queries used to confirm recovery.
