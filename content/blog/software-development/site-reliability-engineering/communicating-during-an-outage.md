---
title: "Communicating During an Outage: The Half of Incident Response Nobody Trains For"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "An outage is a technical problem for the first ten minutes and a communication problem for the rest of it; learn the audiences, the status-page cadence, the comms-lead role, and the exact templates that keep a 50-minute incident from becoming a reputational one."
tags:
  [
    "site-reliability-engineering",
    "sre",
    "incident-communication",
    "incident-response",
    "status-page",
    "comms-lead",
    "on-call",
    "outage",
    "operations",
    "trust",
  ]
category: "software-development"
subcategory: "Site Reliability Engineering"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/communicating-during-an-outage-1.png"
---

At 14:02 on a Tuesday, the checkout API started returning 503s. By 14:05 the on-call engineer had a hypothesis, by 14:11 she had rolled back the bad deploy, and by 14:23 error rates were back to baseline. Twenty-one minutes, wall to wall. A clean incident, technically.

It was also the worst day that team had with their customers all year — and the engineering work had nothing to do with it. For twenty-one minutes, the company said *nothing*. The status page stayed green. Support had no script, so they told the first hundred customers "we're not aware of any issues" — which was a lie, just an uninformed one. By the time checkout recovered, the company's biggest customer had posted a screenshot of the failed payment to a thread that got two thousand replies, a competitor's sales rep had slid into that thread, and the support queue had a four-hour backlog that took two days to clear. The outage lasted twenty-one minutes. The damage lasted two weeks. And not one minute of that damage was technical.

This is the half of incident response nobody trains for. We drill the rollback, the failover, the runbook, the bridge call. We measure MTTR — mean time to recovery, the average wall-clock minutes from detection to resolution — and we optimize it ruthlessly. Then we let the *communication* during those minutes happen by accident: whoever has the most adrenaline grabs the status page, types something defensive, and we wonder why customers don't trust us. The thesis of this post is simple and it is the whole post: **communicate early, on a clock, honestly but bounded, to the right audiences.** Silence is not neutral. Silence is a message, and the message it sends is "we don't know, or we don't care." Customers fill silence with the worst story they can imagine, and that story becomes the outage.

![A vertical stack showing the three outage audiences with their distinct needs from internal responders to stakeholders to external users, all owned by a single comms lead so the incident commander can think](/imgs/blogs/communicating-during-an-outage-1.png)

By the end of this post you will be able to do five concrete things: name your three audiences and write the right message for each; run a status page on a severity-based clock so a customer is never staring at a green dashboard during a red incident; staff the comms-lead role so your incident commander never touches the keyboard; fill in four copy-and-adapt templates (status-page update, internal sitrep, stakeholder update, public postmortem) live during a Sev1; and recognize the silence failure mode early enough to break it. This sits inside the series' loop — define reliability, measure it, spend the error budget, respond to incidents, learn, engineer the fix — squarely in the *respond* phase, alongside [incident command](/blog/software-development/site-reliability-engineering/incident-command-staying-calm-under-fire) and [the anatomy of an incident](/blog/software-development/site-reliability-engineering/the-anatomy-of-an-incident). If you want the why-this-all-matters framing, start at [reliability is a feature](/blog/software-development/site-reliability-engineering/reliability-is-a-feature-the-sre-mindset). This is the chapter that says: getting the bytes flowing again is necessary, and it is not sufficient.

## 1. The principle: silence is the loudest thing you can broadcast

Let's establish the foundational reasoning before any template, because every template downstream is just an expression of one idea: **an information vacuum does not stay empty.**

When your service breaks, your customers find out before you finish your status-page draft. Their app spins, their payment fails, their dashboard 500s. At that instant they have a question — "is it me or is it them?" — and they will get an answer one way or another. If you answer it, the answer is yours: bounded, accurate, calming. If you stay silent, the answer comes from somewhere else: from a competitor, from a frustrated power user on social media, from the support agent guessing, from the customer's own catastrophizing imagination. The information will flow. The only variable you control is whether *you* are the source.

This is why the cost of a comms mistake is asymmetric. A technical mistake during an incident — a rollback that takes two tries, a failover that's slower than the runbook claimed — costs you minutes. A comms mistake costs you trust, and trust is the slowest thing on earth to rebuild. A customer will forgive an outage; outages happen to everyone and they know it. What they will not forgive is the *feeling* that you were hiding it, that you didn't care, or that you lied about it. Notice the word "feeling." Customers do not have access to your incident channel, your bridge call, or your heroic rollback. They have access to exactly one thing: what you tell them. Their entire model of how you handled the incident is built from your communication. You can run a flawless technical response and still be remembered as the company that "went dark for an hour."

There is a control-systems way to see this. A system with no feedback signal does not sit at neutral; it diverges, because every observer extrapolates from the last data point they had using their own priors — and during an outage, everyone's prior is fear. The customer who saw a failed payment and then heard nothing for forty minutes does not conclude "they've probably got it handled." They conclude "my money is gone and nobody's home." The support agent with no information does not say "let me find out." They improvise, and improvisation under pressure trends negative. Silence is a high-gain amplifier pointed at everyone's worst assumption.

It helps to separate two things that we usually bundle into the word "communication," because they decay at very different rates. The first is *information*: the actual facts of what's broken and what you're doing. The second is *signal of presence*: the simple message that a competent human is awake, aware, and engaged. Information has a long shelf life — a customer who learns at T+3 that you've identified the problem can coast on that for a while. But signal-of-presence decays *fast*; it has a half-life measured in minutes. The reassurance from your last post fades steadily, and by the time fifteen minutes have passed with nothing new, the customer's felt sense is back to "is anyone even there?" even though the *information* hasn't changed at all. This is the precise reason the cadence matters even when there's no news to share: you're not re-transmitting information, you're refreshing the signal-of-presence before it decays to zero. A "no change, still on it" update carries no new bits and yet does its entire job, because its job was never the bits. Once you internalize that communication is two separable channels — slow-decaying facts and fast-decaying presence — the whole discipline of "update on a clock even with nothing new" stops feeling like theater and starts feeling like maintenance of a signal that's actively fading.

So the first principle, before cadence and templates and roles, is this: **post early, even when you have nothing.** "We are investigating reports of errors with checkout" — published at minute three, before you know the cause, before you know the ETA — is worth more than a perfectly-worded root-cause analysis published at minute fifty. The early post is not informative in the technical sense; you've told them almost nothing. But it does the one thing that matters: it closes the vacuum. It says *a human knows, a human is on it, and that human will keep talking to you.* Everything else in this post is mechanics for keeping that promise.

The contrast is stark enough to draw. The before-and-after figure below puts the silent path next to the on-a-clock path for the same outage: fifty minutes of zero posts floods support five-fold and lets social media write your story, while five posts with a fifteen-minute maximum gap deflect tickets and hold trust. Same incident, same downtime, opposite outcome — and the only difference is whether you spoke.

![A two-column comparison of going silent for fifty minutes against posting on a clock every fifteen minutes, with the silent path leading to flooded support and a social-media vacuum and the clock path leading to deflected tickets and preserved trust](/imgs/blogs/communicating-during-an-outage-3.png)

#### Worked example: the cost of forty minutes of silence

Take the same 50-minute outage two ways. In the **silent version**, the company posts nothing until the incident is resolved at T+50, then publishes a polished "we experienced a brief service disruption" note. In the **on-a-clock version**, the company posts at T+3, T+12, T+25, T+38, and T+50.

Assume the service has 40,000 active users during the window, and that a steady 3% of affected users will open a support ticket if they're left in the dark, but only 0.4% will if there's a clear status-page post they can find. The silent version generates roughly $40{,}000 \times 0.03 = 1{,}200$ tickets. The on-a-clock version generates roughly $40{,}000 \times 0.004 = 160$ tickets. That is not a rounding difference — it is a 7.5x difference in support load, and support load during an incident is the thing that turns a contained problem into an all-hands scramble where engineers get pulled off the fix to draft replies. The numbers here are illustrative — your real deflection rate depends on your audience and how findable your status page is — but the *direction* and order of magnitude are robust and well-documented across status-page vendors: a visible, updated status page deflects the large majority of would-be tickets. Silence does the opposite. It manufactures load at exactly the moment you have the least capacity to absorb it.

## 2. Three audiences, three completely different messages

The single most common comms failure is treating "communicating during an outage" as one task with one message. It is three tasks. Three audiences, who need different facts, in different language, at different speeds, through different channels. Send the wrong message to the wrong audience and you do harm: dump raw stack traces on customers and you look chaotic and leak architecture; send marketing-smooth "we're committed to excellence" pablum to your responders and you slow them down and insult them.

The first figure above lays out the three audiences as a stack, each feeding the one below, all owned by a single comms lead. Let's take them one at a time.

**Internal responders** — the engineers, the incident commander, the on-call across dependent teams. They need the **technical truth, fast, with no smoothing**. "auth-service p99 is 4.2s, error rate 18%, started 14:01 right after deploy v2.8.1, rolling back now, ETA on rollback 4 min." Jargon is *good* here; it's precise. This conversation lives in one place — the incident channel (a Slack/Teams channel, a bridge call, or both) — and it is owned by the scribe and the IC. The failure mode here is *fragmentation*: three side-threads, a DM to the database team, a separate call the IC doesn't know about. We'll fix that in section 7.

**Stakeholders** — leadership, support leads, account managers, sales, sometimes legal. They are not going to read PromQL and they should not have to. They need **impact, ETA, and what's being done, in business language**. "Checkout is down for all users. We've identified a bad deploy and are rolling back; we expect recovery within 15 minutes. Customers cannot complete purchases. Support: please use the macro in the pinned message. Next update at 14:25." Notice what's in there: who's affected (all users), what's broken (checkout, i.e. revenue), what we're doing (rollback), a time-bound expectation, an instruction for support, and a commitment to update. Notice what's *not* in there: which microservice, which engineer, the stack trace, speculation about the cloud provider. Stakeholders translate this to *their* audiences — the account manager calls the big customer, support arms the queue — so the message has to be repeatable by a non-engineer without distortion.

**External users** — customers, the public, anyone who hits your product. They get the **status page**: an honest acknowledgment of impact, a commitment to a next update, and *nothing else*. No root-cause speculation. No ETA you can't hit. No blame, especially not on a named vendor. The external message is the most constrained of the three precisely because it's the one you can least take back. An internal message that's slightly wrong gets corrected in the next sentence by someone who was there. A public message that's wrong gets screenshotted.

Here's the trap that catches good engineers: the instinct that honesty means *full disclosure to everyone*. It does not. Honesty means **don't lie to anyone** and **don't say more than you know to the people who'll act on it as gospel**. "We're investigating elevated error rates affecting checkout" is fully honest and appropriately bounded for the public. "It's the new connection-pool config in the payments service, we think" is honest to your responders and *reckless* to your public, because you might be wrong, and now you've publicly accused your own (correct) code or, worse, a third party.

| Audience | Channel | What they need | Language | Cadence | Cardinal sin |
| --- | --- | --- | --- | --- | --- |
| Internal responders | Incident channel / bridge | Raw technical truth, live | Jargon-precise | Continuous | Fragmenting into side-threads |
| Stakeholders | Email / leadership channel | Impact + ETA + actions | Business, no jargon | Every 15–30 min | Burying the impact under detail |
| External users | Public status page | Honest acknowledgment + next-update time | Plain, calm, bounded | Severity-driven | Speculating on cause or ETA |

If you remember one thing from this section: the same fact gets *three different sentences*. "Connection pool exhausted in payments-svc after v2.8.1" (internal) becomes "Checkout is down due to a bad deploy; rolling back, ~15 min" (stakeholder) becomes "We're investigating issues affecting checkout and are working on a fix" (external). One incident, one truth, three translations.

Why do the internal and external messages pull in opposite directions on detail? Because they're optimizing for opposite kinds of error. Internally, the cost of *too little* detail is high — a responder who doesn't know the exact error rate or the exact deploy version makes worse decisions — and the cost of *too much* detail is near zero, because the audience can absorb it and self-filter. So you maximize detail internally. Externally, the calculus flips: the cost of too much detail is high — a half-formed cause theory gets quoted back at you, a stack trace leaks architecture, a number you didn't mean as a commitment becomes one — while the marginal value of detail to a customer is low, because all they actually need to decide their next action is "is it you, and are you on it." So you minimize detail externally. The stakeholder message sits in between precisely because stakeholders act *on behalf of* others: they need enough to make business calls and to repeat accurately, but not so much that the repetition distorts. Seeing it as an error-cost optimization, rather than as "be honest vs. be vague," is what keeps you from the two classic mistakes — dumping internal detail on the public, or starving your responders to keep the message "clean."

A second-order point worth making explicit: these audiences talk to each other, and a contradiction between two of your messages is itself a trust-destroying event. If your status page says "a small number of users" while a leaked internal screenshot says "total outage, all users," you've now manufactured a *credibility* incident on top of the reliability one — the gap between your public and private accounts becomes the story. The defense is that all three messages descend from one source of truth (the incident channel), translated by one owner (the comms lead). They differ in *altitude and language*, never in *facts*. "Total checkout outage" can become "checkout is currently unavailable" for the public — softer words, same fact — but it can never become "minor checkout issues," because that's a different claim, and the day someone compares the two versions, the softer-words framing survives and the different-claim framing does not.

## 3. The status page: post early, update on a clock, follow the lifecycle

The status page is the public face of your reliability. It is where the world goes when your product feels broken, and the discipline you bring to it is, for most of your users, the *entirety* of your incident response that they will ever perceive. Let's make it run on rails.

**When to post: early. Earlier than feels comfortable.** The rule is to post when you have *confirmed customer impact*, not when you have a diagnosis. You almost never have a diagnosis early, and waiting for one is how you end up posting at T+40. The first post can be — should be — content-light: "We are investigating reports of errors affecting the feature, and we'll provide an update by a stated time." That's it. You have acknowledged impact and committed to a next update. Those two moves close the vacuum. A useful internal gate: if you've declared a Sev1 or Sev2 and a status-page post hasn't gone up, that is itself a process failure, the same as not paging the on-call.

**The cadence: update every N minutes even if there's nothing new.** This is the counterintuitive heart of status-page discipline. The most common objection is "but we have nothing to add, why post 'no change'?" Because **silence reads as absence, not as steadiness.** A customer watching your status page does not know whether the lack of a new update means "still working it, methodically" or "everyone went home." The only way they can tell is if you tell them. An update that says "We're still working to restore checkout. No change since the last update; the team is actively engaged. Next update by 14:40." is not filler — it is the signal that the system is still being attended to by a human who respects the customer's need to know. The cadence is the heartbeat. A missing beat is alarming even if the heart is fine.

The cadence should scale with severity, which is why the second figure here is a severity-cadence matrix — a Sev1 (total outage of a critical flow) warrants a first post within five minutes and an update every fifteen, while a Sev3 (minor, degraded, internal) may not warrant a public post at all. We'll formalize that table in section 6.

**The lifecycle: investigating → identified → monitoring → resolved.** Most status-page tools (Statuspage, Instatus, Better Stack, Cachet, and friends) encode exactly these four states, and they map cleanly to the real arc of an incident, which the timeline figure below traces across our 50-minute example:

- **Investigating** — "We see it, we're on it, we don't yet know why." Acknowledge impact; commit to next update; *no cause*.
- **Identified** — "We know what's wrong and we're fixing it." Name the symptom and who's affected; describe the action ("rolling back," "failing over"); *still no public root-cause theory unless you're certain and it's safe to share*.
- **Monitoring** — "The fix is in; we're watching to confirm." This is a crucial honest state: you've deployed the mitigation but you don't yet declare victory. Customers see you being careful, which builds far more trust than a premature "all clear" that you have to walk back.
- **Resolved** — "Recovery confirmed and holding." A short recap, an apology proportionate to impact, and a promise of a fuller postmortem if the severity warrants it.

![A left-to-right timeline of a fifty-minute outage with status-page posts at three, twelve, twenty-five, thirty-eight, and fifty minutes moving through investigating, identified, monitoring, and resolved](/imgs/blogs/communicating-during-an-outage-2.png)

Each state has a *job*, and the job dictates what you may and may not publish — which is exactly the structure of the tree figure below. While impact is ongoing you sit in investigating or identified (acknowledge, then name the symptom — never the unconfirmed cause); as impact eases you move to monitoring and then resolved (watch carefully, then recap with a postmortem promise). The state you're in tells you what sentence is allowed, so you never have to invent the framing under pressure.

![A decision tree of the status-page lifecycle splitting impact-ongoing into investigating and identified states and impact-easing into monitoring and resolved states, each annotated with what may be published](/imgs/blogs/communicating-during-an-outage-7.png)

**What to say and what not to say.** Say: the impact in plain terms ("customers cannot complete checkout"), the fact that you're engaged, and a next-update time. Do not say: a root cause you haven't confirmed; an ETA for the fix you can't guarantee; anything that blames a named third party; anything that minimizes ("minor blip," "small subset" when it's not small). Two of those deserve their own sections because they're the ones that turn a technical incident into a reputational one — the false ETA (section 8) and the public blame (section 9 war story). But internalize the rule now: **acknowledge impact, commit to a next update, withhold everything you're not certain of.**

#### Worked example: a full 50-minute status-page timeline, word for word

Here is the on-a-clock version of our opening outage, with the actual text you'd publish. This is the artifact — copy the shapes, change the nouns.

```yaml
# Status-page timeline — checkout outage, illustrative wording
T+3m  [INVESTIGATING]
  "We're investigating reports of errors affecting checkout. Some
   customers may be unable to complete purchases. We're actively
   looking into it and will post an update by 14:18."

T+12m [IDENTIFIED]
  "We've identified the cause of the checkout errors and are deploying
   a fix. Customers may still see failed purchases during this time.
   We'll update again by 14:30."

T+25m [INVESTIGATING / no-change update]
  "The fix is taking longer than expected. Our team remains fully
   engaged on restoring checkout. There is no customer action needed.
   Next update by 14:42."

T+38m [MONITORING]
  "We've deployed a fix and checkout is recovering. We're monitoring
   closely to confirm full recovery before we mark this resolved.
   Next update by 14:55."

T+50m [RESOLVED]
  "Checkout has been fully restored as of 14:52. We're sorry for the
   disruption to your purchases. We'll publish a detailed incident
   report within 48 hours. Thank you for your patience."
```

Look at what the customer experienced: never more than fifteen minutes of silence, always a next-update time they could check their watch against, an honest "taking longer than expected" instead of a fake all-clear, a careful monitoring step, and a resolution with an apology and a promise of follow-up. Now compare the silent version: a green status page for fifty minutes, then a single line at the end — "We experienced a brief service disruption and have since resolved it." That sentence is *technically* true and it is a betrayal, because for fifty minutes the customer's reality (failed payments) and your published reality (all systems operational) were in open contradiction, and they noticed. The word "brief" in particular is the kind of minimization that reads as gaslighting to someone who just lost a sale. The two timelines describe the identical 50-minute technical event. One built trust. One destroyed it.

## 4. The comms lead: a dedicated role so the IC can think

Here is a mistake I have watched bright people make in every incident they ran before they learned this lesson: the incident commander writes the status updates. It seems efficient — the IC has the most context, so the IC should communicate, right? Wrong, and dangerously so.

The reason ties directly to [incident command](/blog/software-development/site-reliability-engineering/incident-command-staying-calm-under-fire). The IC's entire value is holding the *whole* picture and coordinating the response — deciding the next action, delegating it, tracking what's in flight, knowing when to escalate, knowing when to mitigate-first and diagnose-later. That job is a full-time cognitive load, and it is *fragile*. Every time the IC stops coordinating to wordsmith a customer-facing sentence, the response stalls. Worse, comms work is exactly the kind of context-switch that's expensive: writing for the public means switching from "what's the next technical move" to "how does this read to a frightened customer," which are different brains. An IC who's drafting the status page is, for those ninety seconds, not commanding the incident. Do that five times in an hour and you've got an incident with no one at the wheel for eight minutes total, scattered across the worst moments.

So you split it. The **comms lead** (sometimes "communications coordinator" or "scribe-plus") is a distinct role, activated for any incident above a threshold severity, whose entire job is the three-audience translation from section 2. The comms lead:

- Reads the incident channel as the single source of truth and pulls facts from it — they do **not** interrupt responders to ask "what should I tell customers," they listen and translate.
- Drafts and posts status-page updates on the severity cadence.
- Writes and sends the stakeholder updates.
- Arms support with a canned reply (a "macro").
- Owns the next-update clock — when the timer's about to fire, *they* post, even if it's a no-change update, without the IC having to remember.
- Surfaces to the IC only the comms-relevant decisions: "Leadership is asking for an ETA — what can I commit to?" or "A reporter just emailed; routing to PR."

The graph figure below shows the flow: the IC and scribe maintain one source of truth, the comms lead fans it out to stakeholders and the public status page in parallel and arms support, and the result is held trust with zero rumor side-threads. The IC's keyboard never touches the status page. That separation is the whole point.

![A branching flow diagram showing the incident commander and scribe as one source of truth feeding a comms lead who fans out to stakeholder updates and the public status page and arms support with a canned reply leading to preserved trust](/imgs/blogs/communicating-during-an-outage-5.png)

**Who plays comms lead?** Not necessarily an engineer. In a mature org it's often a dedicated incident-management person, a TPM, a support lead, or an SRE who isn't hands-on-keyboard for this particular incident. The key qualifications are: can write clearly under pressure, knows the product's customer-facing language, and has the authority (pre-granted) to post to the status page without a sign-off loop. That last point is critical: if every status-page update needs a VP's approval, you will miss your cadence every time, because the VP is in a meeting. Pre-authorize the comms lead to post within a template. The template *is* the guardrail; that's why the templates in the next section exist.

**The proof this works:** before splitting the role, the teams I've seen typically post their first status-page update at T+15 to T+25 (the IC gets to it once the technical fire is contained) and miss their cadence repeatedly during the incident. After splitting it, first-post times drop to T+3 to T+5 and cadence adherence goes to essentially 100%, because someone owns the clock and only the clock. The IC's coordination quality goes up too, measurably, because they stop dropping the response thread to write. You don't need a study to feel this; run one incident each way and you'll never go back.

### Measuring comms quality honestly

The same series that tells you to measure reliability as an SLI — a ratio of good events over a rolling window — gives you the tools to measure comms, and you should, because "the comms felt bad" is not actionable in a postmortem and "we posted 22 minutes late" is. Treat communication as a set of incident-level metrics you pull from your incident records and status-page timestamps:

- **Time to first public post (TTFP):** declaration timestamp to first status-page post. This is your comms equivalent of time-to-detect. Target by severity (≤5 min Sev1). Track the distribution, not just the mean — the tail is where the silence cascades live.
- **Cadence adherence:** the fraction of inter-update gaps that came in under the severity target. A Sev1 with five updates and one 22-minute gap is at 80% adherence; the 20% miss is exactly the window where trust leaked.
- **Maximum silence window:** the single longest gap with no public post during the incident. This is the most predictive single number for reputational damage, because customers experience the *worst* gap, not the average.
- **Ticket-deflection ratio:** support tickets opened during the incident divided by affected users, compared against your baseline. A visible, updated status page should pull this far below the "dark" rate.
- **Correction count:** how many times you had to walk back a public statement (a premature all-clear, a wrong cause, a missed ETA). The target is zero; every correction is a credibility withdrawal.

#### Worked example: comms before and after, by the numbers

A real-shaped before/after for a team that adopted the comms-lead role and severity cadence. Numbers are illustrative but the shape and magnitude track what teams actually see:

| Metric | Before (no role, ad-hoc) | After (comms lead + clock) |
| --- | --- | --- |
| Time to first public post (Sev1) | 18 min | 4 min |
| Cadence adherence (Sev1) | ~40% | ~100% |
| Max silence window | 31 min | 14 min |
| Tickets per 1,000 affected users | ~30 | ~4 |
| Public corrections per incident | ~1.2 | ~0.1 |

Read the table as a single story: by *only* changing who communicates and on what clock — not the engineering, not the MTTR — the team cut its worst silence window by more than half, dropped support load by roughly 7x, and nearly eliminated the public walk-backs that are individually the most trust-corrosive comms events. None of those gains required a faster fix. They're the pure return on communication discipline, and they're measurable, which means they're improvable: pick the metric with the ugliest tail (usually max silence window), and engineer it down the same way you'd engineer down a latency p99.

## 5. The templates: structure so you don't have to think while the building burns

Under stress, working memory collapses. The single best gift you can give your future self at T+3 of a Sev1 is a fill-in-the-blank template, because it converts "what do I even say" (a creative task, hard under adrenaline) into "fill these four slots" (a recall task, easy). Every template below has the same backbone, the **four slots that every outage update needs**:

1. **What's happening** — the impact, in the audience's language.
2. **Who's affected** — scope. All users? A region? A feature?
3. **What we're doing** — the action, at the audience's altitude.
4. **Next update at HH:MM** — the commitment. Never omit this; it is the heartbeat.

### The status-page update template

```yaml
# Public status-page update — fill the [slots]
[STATE: Investigating | Identified | Monitoring | Resolved]

We're [investigating | aware of | fixing] an issue affecting
[feature/area, e.g. "checkout"]. [Optional: scope — "Some users in
the EU region" or "All users"] may experience [observed symptom in
plain terms — "errors when completing a purchase"].

[If Identified+: "Our team has [action — 'identified the cause and is
deploying a fix' | 'failed over to a healthy region']."]

There is no action needed on your part. / [If action needed: state it.]

We'll provide the next update by [HH:MM timezone].
```

### The internal sitrep template

A **sitrep** (situation report) is the periodic internal pulse — posted in the incident channel by the scribe or comms lead so anyone joining the bridge gets oriented in ten seconds without scrolling. It is the antidote to "wait, what's the current status?" being asked by every newcomer.

```yaml
# Internal sitrep — posted in the incident channel every ~10 min
SITREP @ 14:25 — INC-2291 — Sev1 — checkout 503s

STATUS:    Mitigating (rollback in progress)
IMPACT:    Checkout fully down, all users, since 14:01 (~24 min)
HYPOTHESIS: v2.8.1 exhausted the payments-svc connection pool
ACTIONS:   - Rollback to v2.8.0 deploying now (owner: @alex, ETA 4m)
           - DB team verifying pool metrics (owner: @sam)
NEXT:      Confirm error rate < 1% post-rollback
IC:        @priya   SCRIBE: @jordan   COMMS: @lee
NEXT SITREP: 14:35
```

Notice the sitrep names the IC, scribe, and comms lead explicitly. During a chaotic incident, "who's running this?" should be answerable by reading the latest sitrep, not by asking. The sitrep is also where the next-update clock for the *internal* audience lives.

### The stakeholder update template

```yaml
# Stakeholder / leadership update — email or leadership channel
Subject: [Sev1] Checkout outage — IMPACT + ETA — update 2

IMPACT:   Checkout is down for all users since 14:01. Customers
          cannot complete purchases. Estimated revenue exposure is
          being tracked by Finance.
CAUSE:    A bad deploy (internal). No data loss. No security impact.
STATUS:   Rolling back; expected recovery within ~15 minutes.
CUSTOMER: Status page is updated. Support is armed with a macro.
          Account managers for top-20 accounts have been notified.
NEXT:     We'll send the next update by 14:25, or sooner on recovery.
OWNER:    Priya (IC). Questions to the #inc-2291 channel.
```

The stakeholder update leads with **IMPACT** and **ETA** because that's what leadership needs to make their decisions (do we proactively call the big customer? do we hold the press release going out at 3pm?). It explicitly addresses the two questions leadership always has during a customer-facing incident — "is there data loss?" and "is this a security thing?" — even just to say "no," because an unanswered "is our data safe?" escalates faster than the outage itself.

### The public postmortem / incident report template

After resolution, for any significant incident, you publish externally. This is the trust-*builder*, the inverse of the silence that's the trust-destroyer. A good public incident report says: here's what happened, here's the real impact, here's what we got wrong, and here's specifically what we're changing so it doesn't recur. It is blameless toward individuals (see [the blameless postmortem](/blog/software-development/site-reliability-engineering/the-blameless-postmortem), the sibling post on running the internal version) but accountable as an organization.

```yaml
# Public incident report — published within 48–72h of a major incident
TITLE:    Checkout outage on 2026-06-16 (50 minutes)

SUMMARY:  On June 16 from 14:01 to 14:52 UTC, customers were unable to
          complete checkout. The cause was a configuration change that
          exhausted a database connection pool. We have since fixed
          the change and added a safeguard. We're sorry.

IMPACT:   All checkout attempts failed for 50 minutes. No data was
          lost or exposed. Failed payments were not charged.

TIMELINE: 14:01 — error rate spiked after a deploy
          14:04 — alerted, incident declared
          14:12 — cause identified
          14:23 — rollback complete, recovery began
          14:52 — fully restored and confirmed

WHAT WENT WRONG (system, not people):
          - The config lacked a connection-pool ceiling.
          - Our canary didn't exercise peak checkout load.

WHAT WE'RE CHANGING:
          - Enforce a pool ceiling in config validation (done).
          - Add a load step to the checkout canary (in progress).
          - Tighten the status-page SLA to a first post within 5 min.
```

The public report follows the same accountability tone as the internal postmortem but strips the names — you describe *system* failures, never "engineer X pushed a bad config." Customers don't need a scapegoat; they need confidence you understand your own system and have made it safer. Publishing a clear, specific, un-defensive incident report after a bad outage is one of the highest-leverage trust moves available to an engineering org. It's also the best advertisement for your competence: "they were transparent and they actually fixed it" is a story customers tell *for* you.

## 6. Cadence by severity: a clock you don't have to argue about at 2am

Cadence shouldn't be a judgment call you make mid-incident, because mid-incident judgment is exactly what's degraded. Decide it once, in calm daylight, and encode it in your incident runbook as a table tied to your severity definitions. The matrix figure below renders this as severity rows against first-post, cadence, and audience columns; here it is as the runnable reference your on-call can pull up at 2am.

![A severity-by-cadence matrix table mapping Sev1, Sev2, and Sev3 to their required first-post time, update cadence, and audience scope from public-and-all down to responders-only](/imgs/blogs/communicating-during-an-outage-4.png)

| Severity | What it means | First status-page post | Update cadence | Audience | Stakeholder updates |
| --- | --- | --- | --- | --- | --- |
| **Sev1** | Critical flow down, all/most users | within 5 min | every 15 min | Public + leadership + all | every 15–30 min |
| **Sev2** | Major degradation or partial outage | within 15 min | every 30 min | Public if customer-visible | every 30–60 min |
| **Sev3** | Minor / degraded / internal-only | optional (post if visible) | every 60 min if posted | Internal; public only if impact | as needed |

A few notes that keep this honest. First, the "first post" clock starts at **incident declaration**, not at "when we understand it" — that's the whole point, you post before you understand. Second, the cadence is a *ceiling on silence*, not a floor on noise: post sooner whenever there's real news (a state change, recovery, a worsening). The clock is there to guarantee you never *under*-communicate; it never stops you from communicating more. Third, "optional" for Sev3 is a real option — not every degraded internal batch job needs a public post, and posting trivia trains customers to ignore your status page, which is its own failure mode. Reserve the public page for things customers can actually perceive.

#### Worked example: turning severity into a wall-clock plan

A Sev1 declared at 14:00 generates a fixed, knowable comms schedule the moment it's declared, before anyone knows the cause:

- 14:00 — incident declared, comms lead activated.
- by 14:05 — first status-page post (Investigating). Stakeholder update 1 sent.
- 14:15, 14:30, 14:45, 15:00… — status-page updates every 15 min, each carrying the next-update time, until Resolved.
- Stakeholder updates ride along every other status-page beat (every 30 min) unless leadership asks for more.

The power of writing it as a clock is that the comms lead doesn't *decide* anything about timing during the incident — they execute a schedule. Decisions consume the scarce resource (attention); schedules don't. A Sev2 declared at the same time gets the same shape with the dials turned down: first post by 14:15, updates every 30 min. The dials are the only thing severity changes; the *discipline* is constant.

## 7. Internal comms discipline: one channel, one source of truth, no rumor side-threads

External comms gets the glory in this post, but internal comms discipline is where incidents are quietly won or lost, and it's where most teams are sloppiest. The failure mode is fragmentation, and it's insidious because each fragment feels helpful in the moment.

Picture the anti-pattern. The incident is in `#inc-2291`. But the database expert is debugging in a DM with the on-call. Two engineers spun up a separate "quick sync" call the IC doesn't know about. A manager is asking for status in a *different* channel, `#leadership`, and getting answers from someone who isn't in the incident channel and is half a step behind. Now there are four versions of "the current status" in four places, three of them stale, and the IC's mental model is being corrupted by people acting on the wrong one. Someone reports "the rollback worked!" in the side-call; the IC, hearing nothing, keeps the team on the rollback; ten minutes evaporate.

The discipline that fixes this is a set of hard rules, and they're worth pinning to the top of every incident channel:

- **One channel is the source of truth.** All incident comms happen in the incident channel. If a side conversation must happen (a deep debug, a vendor call), its *outcome* gets posted back to the main channel immediately. The channel is the system of record; if it's not in the channel, it didn't happen.
- **The IC/scribe owns the status.** "The current status" is whatever the latest sitrep says, full stop. Nobody else declares status. This kills the "I heard the rollback worked" rumor — it's not status until the scribe writes it in a sitrep after the IC confirms it.
- **No rumor side-threads.** Speculation is fine *in the channel* where the IC can see it and weigh it; speculation in DMs and side-calls is poison because it acts on the response invisibly. "I bet it's the cache" is a useful hypothesis when it's in the channel and a landmine when it's a DM that someone acts on alone.
- **Regular sitreps.** Every ~10 minutes (tighter for fast-moving Sev1s), the scribe posts the sitrep from section 5. This is the internal heartbeat, and it does for responders what the status page does for customers: it lets anyone — a newly-paged engineer, a curious VP lurking in the channel — get current in ten seconds without interrupting the people working.

The deep reason the single-channel rule matters connects back to the IC's job. The IC coordinates based on a model of reality assembled from what they can see. Every fragment of incident activity that happens where the IC *can't* see it is a place where the IC's model and reality diverge — and the IC makes decisions from the model. Fragmentation isn't just messy; it actively corrupts the command function. One channel isn't tidiness for its own sake; it's keeping the commander's map matched to the territory.

There's a tension worth naming. Forcing everything into one channel can make the channel noisy — fifty messages of debug chatter burying the three that matter. The resolution is *threads and roles*, not *more channels*: use the platform's threading for deep dives (a debug thread hangs off the message it relates to, keeping the main flow scannable), and let the scribe's sitreps be the canonical scannable summary so nobody has to read all fifty messages to get current. One channel, threaded for depth, with a periodic sitrep as the executive summary. That scales to a surprisingly large incident before you need to split, and when you do split (a genuinely huge multi-workstream incident), you split by *workstream with a lead per stream reporting up to the IC*, never by ad-hoc convenience.

### Handoffs and the long incident

Most outages resolve in under an hour, but the ones that hurt most run for hours, and a long incident introduces a comms failure mode the short one never does: **fatigue and handoff drift.** The comms lead who's been writing updates for three hours is degraded — slower, terser, more likely to miss the cadence clock or to let frustration leak into the tone. So bake handoff into the role. After roughly ninety minutes to two hours, hand the comms lead to a fresh person, and do it *explicitly and in-channel*: "Comms handoff — @lee is rolling off, @morgan is now comms lead. Morgan has read the timeline and the last three updates. Next public update due 16:45." The incoming comms lead inherits three concrete artifacts: the current status (latest sitrep), the next-update clock (so the cadence doesn't reset or skip), and the running list of what's been said publicly (so they don't contradict an earlier post). The handoff that *isn't* explicit is where you get the classic long-incident comms break: the clock resets silently, a thirty-minute gap opens during the transition, and a customer who'd been getting reliable updates suddenly hits silence at the worst time. Treat a comms handoff exactly like a code deploy — announced, logged, with a clear before-and-after owner — because an unannounced handoff is a silent config change to your most customer-facing system.

Now stress-test the whole internal model against the hard cases. *What if two incidents overlap?* Then you need two incident channels and two ICs, and a single comms lead can rarely serve both — split comms too, and have someone (often an incident-manager-on-call) deconflict so you don't post two contradictory status updates or let one incident's cadence starve the other. *What if the on-call who declared it goes dark — laptop dies, falls asleep on a 4am page?* The sitrep cadence is your safety net: when no sitrep lands at its scheduled time, that *itself* is an alarm that escalates, the same as a missed heartbeat from a service. The discipline that everything lives in one channel with a clock means the *absence* of an expected update is detectable, which means a dropped responder gets noticed in minutes, not at the next shift change. *What if the incident channel itself is part of the outage* — your chat tool is down, or it runs on the infrastructure that's failing? This is why mature incident plans designate an out-of-band fallback (a phone bridge, a separate vendor's chat, an SMS tree) decided *before* the incident; discovering mid-Sev1 that your only comms channel is also down is a uniquely bad five minutes, and the only fix is to have rehearsed the fallback.

## 8. Honest but bounded: say what you know, never an ETA you can't hit

We arrive at the most expensive comms mistake there is, and it's worth its own section because well-meaning people make it constantly out of a desire to be *helpful*: the false ETA.

The instinct is generous. A customer is hurting, you want to comfort them, so you reach for the most comforting thing — a near-term fix time. "We expect to be back within 10 minutes." It feels kind. It is, in fact, a loaded gun pointed at your own credibility. Because here's the arithmetic of trust around a promise: when you make a public, time-bound commitment, you create a checkpoint. At that checkpoint, exactly one of two things happens — you hit it (small trust gain) or you miss it (large trust loss). And the asymmetry is brutal: hitting an ETA you set buys you a little goodwill, but *missing* one detonates trust, because now you've demonstrated two bad things at once — you didn't actually understand the problem (you didn't know how long it'd take) and your word doesn't hold (you said 10, it's 30). The expected value of a fix-ETA you're not certain of is strongly negative.

So what do you promise instead? **The next update.** "We'll provide the next update by 14:30" is a promise entirely within your control. You don't have to fix anything to keep it — you just have to *post*, which the comms lead is going to do anyway. It's a promise you can always keep, and every kept promise compounds: by the third or fourth update that landed exactly when you said it would, the customer has learned that your word is good, which is *worth more than a fast fix*. A customer who trusts that you'll keep talking to them on schedule will wait patiently through a long outage. A customer who caught you missing a fix-ETA stops believing anything you say for the rest of the incident — and they refresh anxiously and open a ticket and post angrily, the exact behaviors the status page exists to prevent.

The before-and-after figure below contrasts the two framings: the over-promised "back in 10 minutes" path that runs through a blown deadline to collapsed trust, versus the "next update in 20 minutes" path that runs through a kept promise to compounding credibility.

![A two-column before and after comparison showing an over-promised back-in-ten-minutes message leading to a blown deadline and collapsed trust versus a next-update-in-twenty-minutes message leading to a kept promise and compounding credibility](/imgs/blogs/communicating-during-an-outage-6.png)

There's a subtle, *correct* version of communicating timing that isn't a false ETA: communicating timing you're genuinely confident in, with appropriate hedging. "We've completed the rollback and expect full recovery within a few minutes" — said while you're in the Monitoring state and watching the error rate actually fall — is fine, because it's grounded in observed reality and hedged ("a few minutes," not "by 14:33:00"). The line is: **describe the state of the world you can observe, not a prediction you can't back up.** "Rolling back now" (observable) good. "Fixed in 10 minutes" (prediction) bad. "Monitoring recovery, looking good" (observable) good. "Should be resolved shortly" (vague, observable-ish) acceptable. The danger zone is always the specific future commitment about something you don't control.

#### Worked example: the over-promise that cost more than the outage

A real-shaped scenario, anonymized and illustrative. A team hits a database failover problem. At T+8 the on-call, wanting to reassure, posts "We're failing over to our backup and expect to be back within 10 minutes." The failover hits an unexpected replication lag issue. At T+18 — the deadline blown — they post nothing (they're heads-down). At T+25 they post "Still working on it." At T+40 they recover. Now reconstruct the customer's experience: at T+18 they're refreshing, the promised time has passed, and there's *silence*. The silence after a missed promise is the worst possible state — it confirms the customer's new theory that you have no idea what you're doing. The technical outage was 40 minutes. But the trust damage was concentrated entirely in the 7-minute window between the blown 10-minute promise and the next post, because that's where the customer's model flipped from "they've got this" to "they're in over their heads." Had the first post said "We're working to restore service and will update you by T+18 minutes," and then at T+18 said "Still working; next update by T+28 minutes," the identical 40-minute outage produces a customer who's mildly annoyed instead of one who's lost faith. Same engineering. Same downtime. Opposite trust outcome — decided entirely by one over-eager sentence at T+8.

## 9. The silence failure mode, and a war story about blaming the vendor

We've circled the silence failure mode throughout; let's name its full anatomy, because recognizing it early is what lets you break it. When you go dark during a customer-visible incident, a predictable cascade fires:

1. **Customers assume the worst.** No data point means extrapolate from fear. "My payment failed and they're silent" becomes "my money is gone / they've been hacked / they're going under."
2. **Support is flooded.** Every anxious customer who can't find an answer on your (still-green) status page opens a ticket or calls. Support, with no script and no information, either guesses (badly) or says "we're not aware of any issue" (a confidence-destroying lie). The ticket queue balloons — recall the 7.5x from section 1 — and now engineers get pulled off the fix to help drain it.
3. **Social media fills the vacuum.** Public threads form. A power user posts a screenshot. Others pile on. The narrative — "company X is down and saying nothing" — sets and hardens *before* you say your first word, so your eventual statement is now fighting an established story instead of writing the first draft of it.
4. **The story outlives the outage.** The technical incident ends; the "they went dark on us" reputation doesn't. This is the asymmetry from section 1 in its final form: minutes of silence, weeks of consequence.

The break is mechanical and you already have it: the severity-cadence clock plus the comms lead who owns it. The clock guarantees a post within five minutes (Sev1); the comms lead guarantees someone whose only job is keeping that promise. Silence becomes a *process violation* you'd catch in a postmortem, not a default you slide into.

### War story: the public vendor-blame that aged badly

This pattern has burned enough companies that it's worth telling as a composite, illustrative war story — the specifics are anonymized but the shape is real and recurring across the industry. A SaaS company suffers a widespread outage. Early in the incident, an engineer is fairly sure the root cause is their cloud provider's networking layer, and someone — wanting to take the heat off the team — posts publicly: "We're experiencing issues due to a problem with our cloud provider. We're waiting on them to resolve it." It feels honest. It feels like it externalizes the blame helpfully. Two things then go wrong, and they're the two reasons you *never* blame a named vendor publicly mid-incident:

First, **they were wrong about the cause.** The cloud provider's status page was all green; the actual problem was the company's own misconfigured network policy that happened to *look* like a provider issue from the inside. Now the public statement is a false accusation against a named company — embarrassing, and in some contracts a contractual problem, and certainly a credibility hit when they have to walk it back.

Second, even **if they'd been right, it doesn't help the customer and it makes the company look smaller.** The customer doesn't care whose fault it is; they care that *your* product is broken and what *you're* doing about it. "It's the cloud provider's fault" reads to a customer as "we're not in control of our own service" — which is, from the customer's seat, true and unreassuring. The mature message owns the *experience* without owning a cause you can't prove: "We're experiencing a disruption affecting checkout. We're working to restore service and will update by a stated time." If, in the postmortem, the cause genuinely was the provider and it's relevant, *that's* where it can be stated — accurately, with the dust settled, in context. Mid-incident, on the public page, naming a vendor is a strictly losing move: right, it doesn't help; wrong, it's a disaster. The rule writes itself — **own the experience, never assign public blame while the incident is live.**

The do/don't of public outage comms distills to the matrix figure below: acknowledge the impact but don't minimize it; say you're investigating but don't guess the root cause; give a next-update time but don't promise a fix ETA; own the customer's experience but don't blame the vendor.

![A do and don't matrix for public outage communication with rows for acknowledging impact, discussing cause, committing to timing, and assigning blame, each split into the right move and the trust-destroying move](/imgs/blogs/communicating-during-an-outage-8.png)

## 10. The support macro, and rehearsing comms before you need it

Two artifacts close the loop between your incident channel and your customers, and both should exist *before* the incident, drafted in calm daylight: the support macro and the comms rehearsal.

**The support macro** is the canned reply your support team uses the instant an incident is declared, so they never improvise "we're not aware of any issue" again. It mirrors the status-page post but in support-reply voice, and it points the customer back to the status page (the single source of truth) rather than trying to recreate it in a ticket:

```yaml
# Support macro — fired the moment a customer-visible incident is declared
"Thanks for reaching out, and I'm sorry you're hitting this. We're
 aware of an issue affecting checkout right now and our team is
 actively working on it. You can follow live updates here:
 status.example.com. I've logged your report so we can follow up if
 anything specific to your account needs attention. We appreciate
 your patience while we get this fixed."
```

Notice the macro does three things at once: it acknowledges the specific customer's pain (empathy), it routes them to the canonical updates (deflection without dismissal — "follow it here" not "go away"), and it logs the report so nothing falls through. The comms lead's job at incident declaration includes *arming* this macro — telling support "incident declared, use the checkout macro, link is live" — so that support's first reply and your status page go out within the same couple of minutes, telling one consistent story. The failure mode this prevents is the worst kind of contradiction: a status page that says "investigating" while support tells the customer "everything's fine." When those two sources disagree, the customer trusts neither.

**Rehearsing comms** is the part teams skip and then regret. You run game days for the technical response — you should run them for the comms response too, because the comms muscles are exactly the ones that seize up under real pressure. In a comms game day you inject a simulated incident and require the on-call to actually *post* a (clearly-marked-as-drill) status update within the five-minute target, draft a stakeholder update, and arm the support macro — on the clock, against the real templates. The first time a team does this, the times are embarrassing: fifteen minutes to write a two-sentence "investigating" post, because nobody had ever done it under a timer and the template wasn't muscle memory yet. By the third drill, it's under three minutes and the templates feel automatic. That's the entire point: the moment to discover that your status-page login doesn't work, your template has a typo, or nobody knows who's authorized to post is during a drill on a Wednesday afternoon, not during a real Sev1 at 3am. Rehearse the comms exactly as seriously as you rehearse the failover, because in front of customers, the comms *is* the incident response.

A note on **proactive comms for planned work**, since it shares all the same machinery: a planned maintenance window, a known risky migration, or a degraded-mode warning. Here you have the luxury of communicating *before* impact — "Scheduled maintenance on checkout this Sunday 02:00–04:00 UTC; you may see brief interruptions" — which is the easiest trust you'll ever bank. The discipline is the same (clear scope, clear window, a contact), but the cost of getting it wrong is far lower because there's no surprise. The trap is the inverse: don't let a *planned* window run long and silent. If your two-hour maintenance hits hour three, that's now an unplanned incident in customers' eyes, and you owe them the same on-a-clock updates you'd give any outage. A planned window that overruns without comms is a special betrayal, because you promised a time and then went quiet past it — the false-ETA failure from section 8, just scheduled in advance.

## 11. Tone: calm, factual, empathetic, accountable without self-flagellation

Everything above is mechanics — who, when, what. Tone is the texture, and it's where good comms become *trusted* comms. The target tone has four properties, and the skill is holding all four at once.

**Calm.** Your words set the emotional temperature of the incident for everyone reading them. Panicked language ("CRITICAL — everything is down!!!") on a status page makes customers panic; flat, steady language ("We're investigating an issue affecting checkout") makes them trust that the people responding are steady. The status page is not the place for exclamation points. Calm is contagious, and so is its opposite.

**Factual.** Say what you know, attribute uncertainty honestly ("we're still investigating the cause"), and never inflate or deflate. Factual tone is what makes the *next* statement believable — if your "investigating" posts are precise and grounded, your "resolved" post is trusted. Vague or hyped language burns the credibility you'll need at the end.

**Empathetic.** Acknowledge the customer's actual experience. "We know this is disrupting your work, and we're sorry" costs nothing and lands enormously, because it signals you understand there's a human on the other end who just lost a sale or missed a deadline. Empathy is not the same as apology-spiral; one sincere acknowledgment of impact is worth more than five "we deeply sincerely apologize"s, which start to read as performance.

**Accountable without self-flagellation.** This is the hardest balance. You want to own it — "this was our issue, here's what we're doing" — without groveling, and without the opposite failure of defensive minimization. Self-flagellation ("we completely failed you, we're so terrible, we don't deserve your trust") is actually a *self-centered* move: it makes the incident about your feelings of guilt rather than the customer's problem, and it reads as either fishing for reassurance or losing composure. Defensive minimization ("a minor blip affecting a small subset") is the other ditch — it tells the customer their real pain doesn't count. The center line is plain, adult accountability: "Checkout was down for 50 minutes. That's not the reliability you should expect from us. Here's exactly what we've changed." Own it, state the fix, move on. Adults owning a mistake calmly is one of the most trust-building things a company can do; adults groveling or deflecting is one of the least.

A practical tone heuristic: **write the update, then read it as the angriest affected customer.** Does "minor disruption" make them angrier (because it wasn't minor to them)? Does the absence of any "we're sorry" make them feel unseen? Does the panicky phrasing make them more scared? Edit until a hurt customer would read it and think "okay, they get it, they're on it, they'll tell me more soon." That's the target. The comms lead role exists partly so there's *someone with the bandwidth to do that read* — the IC, mid-command, cannot.

## 12. How to reach for this (and when not to)

Comms discipline has a cost: the comms lead is a person not doing hands-on response, the status-page cadence is real labor, the templates and severity table are upfront work to build and maintain. So apply it proportionately.

**Reach for the full machinery when:** you have external customers who can perceive the outage; the incident is Sev1 or Sev2 (customer-visible); your product is something people depend on for work or money; or you operate at a scale where social-media narrative formation is a real risk. For these, the comms lead, the severity cadence, and the templates pay for themselves the first time you avoid a silence cascade.

**Don't over-invest when:** the "outage" is an internal-only tool with five users you can Slack directly — just Slack them, no status page needed. Or it's a Sev3 degradation customers genuinely can't perceive — posting it trains people to ignore your status page (the "boy who cried wolf" failure, but for status pages). Or you're a tiny team pre-product-market-fit where a heavyweight incident-comms process is premature ceremony — start with the *one rule that matters most* (post early on the status page, commit to a next update) and add roles and severity tables as you grow. Don't build a 12-role incident-comms org for a service three people use.

The judgment call is the same one this whole series keeps returning to: match the investment to the impact. A 99.999% comms process for a 99.5% product is waste. But under-investing — having *no* plan for who talks to customers during an outage — is the far more common and far more expensive error, because the silence failure mode is the default you fall into when there's no plan, and it's the one that turns technical incidents into reputational ones. If you're going to err, err toward more communication, sooner. The downside of an extra status-page update is mild; the downside of silence is the war story in section 9.

Stress-test the recommendation against the awkward cases, because that's where comms plans actually fail. *What if the budget is already spent* — you're mid-incident and you know this outage has already blown the month's error budget? It changes nothing about the comms; you still post on the clock, still own the impact, still promise the next update. The budget conversation is for the postmortem and the planning that follows, not for the public, and certainly not as an excuse to communicate less ("we're over budget anyway" is never a reason to go dark). *What if you genuinely don't know if it's a real outage or a false alarm* — the alert fired but you're not sure it's customer-impacting? Lean toward an honest, low-key post if there's any plausible customer impact ("we're investigating reports of intermittent errors and will confirm shortly"), and resolve it quickly if it turns out to be nothing. The cost of a brief, accurate "investigating, turned out to be a false alarm, all clear" is trivial; the cost of having said nothing while a real incident built is the silence cascade. *What if leadership wants to suppress the comms* — someone senior says "don't post anything, it'll spook customers"? This is the moment the pre-authorized comms lead and the written severity policy earn their keep: the policy was decided in daylight precisely so that mid-incident pressure can't override it, and "we post Sev1s within five minutes" is a commitment to customers that a panicked executive shouldn't be able to silently revoke. If the policy needs to change, change it in a calm review, not in the middle of the fire.

And remember where this sits in the loop. Communication is part of *respond*, but it feeds *learn*: the public incident report is the customer-facing face of the [blameless postmortem](/blog/software-development/site-reliability-engineering/the-blameless-postmortem), and the comms timeline (when did we post, did we hit cadence, was the first post late?) is itself a thing you review and improve. A postmortem that only examines the technical timeline and ignores "we didn't post for 22 minutes" is missing half the incident. Treat comms as a first-class part of the response that gets measured, reviewed, and engineered — because it is.

## 13. Key takeaways

- **Silence is the loudest broadcast.** An information vacuum fills with the worst story your customers can dream up. Post early — even "we're investigating" with no cause — to close the vacuum.
- **Three audiences, three messages.** Responders need raw technical truth in one channel; stakeholders need impact + ETA in business language; the public needs an honest, bounded acknowledgment with no cause speculation and no blame.
- **Run the status page on a severity clock.** First post within 5 minutes for a Sev1; update every 15 minutes even if there's no change, because a missing beat reads as abandonment. Follow the lifecycle: investigating → identified → monitoring → resolved.
- **Staff a comms lead.** The IC's value is coordination; every status-page sentence the IC writes is coordination not happening. A dedicated comms lead, pre-authorized to post within templates, owns the clock so the IC can think.
- **Promise the next update, never a fix ETA.** "Back in 10 minutes" is a loaded gun pointed at your credibility; "next update by 14:30" is a promise you can always keep, and kept promises compound into trust.
- **One channel, one source of truth.** Fragmentation into DMs and side-calls corrupts the IC's model of reality. Post side-conversation outcomes back to the channel; let regular sitreps be the canonical summary.
- **Never publicly blame a named vendor mid-incident.** If you're wrong it's a false accusation; if you're right it makes you look out of control. Own the customer's experience; save attribution for the postmortem.
- **Templates beat creativity under stress.** Four slots — what's happening, who's affected, what we're doing, next update at HH:MM — turn a hard creative task into an easy recall task.
- **Tone: calm, factual, empathetic, accountable without self-flagellation.** Own the mistake plainly, state the fix, and move on. Read every update as the angriest affected customer before you post it.
- **The public incident report builds the trust the outage spent.** A clear, specific, un-defensive postmortem is the highest-leverage trust move you have after a bad outage.

## Further reading

- [Reliability Is a Feature: The SRE Mindset](/blog/software-development/site-reliability-engineering/reliability-is-a-feature-the-sre-mindset) — why reliability, including how you communicate about it, is a product feature you engineer.
- [Incident Command: Staying Calm Under Fire](/blog/software-development/site-reliability-engineering/incident-command-staying-calm-under-fire) — the IC role this post offloads comms from, and why the separation matters.
- [The Anatomy of an Incident](/blog/software-development/site-reliability-engineering/the-anatomy-of-an-incident) — the full detect → mitigate → resolve arc that the comms timeline runs alongside.
- [The Blameless Postmortem](/blog/software-development/site-reliability-engineering/the-blameless-postmortem) — the internal learning ritual whose public-facing face is the incident report template here.
- *Google SRE Book*, "Managing Incidents" and *SRE Workbook*, "Incident Response" — the canonical treatment of the incident-command and communication roles.
- *Atlassian Incident Management Handbook* and the *PagerDuty Incident Response* documentation — practical, vendor-grounded guides to status-page cadence, comms-lead duties, and stakeholder updates.
- Your status-page tool's own incident-template docs (Statuspage, Instatus, Better Stack, Cachet) — encode the lifecycle states and your severity cadence directly into the tool so the discipline is the path of least resistance.
