---
title: "Designing a Humane On-Call: Rotations, Escalation, and a Pager People Can Live With"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "On-call is a system you design, not a punishment you endure: build a sustainable rotation, an escalation chain that never drops a page, a page budget that flags an unsustainable shift, and a health scorecard you actually act on."
tags:
  [
    "site-reliability-engineering",
    "sre",
    "on-call",
    "incident-response",
    "alerting",
    "burnout",
    "escalation",
    "rotation",
    "observability",
  ]
category: "software-development"
subcategory: "Site Reliability Engineering"
author: "Hiep Tran"
featured: true
readTime: 58
image: "/imgs/blogs/designing-a-humane-on-call-1.png"
---

The best engineer I have ever worked with quit over a pager. Not over money, not over a bad manager, not over a project that got cancelled. She quit because for fourteen months she had been one of three people carrying the pager for a service that paged about thirty times a week, and the math of that had ground her down to nothing. One week in three she slept with her phone face-up on the nightstand, brightness at maximum, ringer at maximum, braced for the buzz. She told me, on her last day, that she had stopped being able to tell the difference between being on-call and being off-call, because even in her clear weeks she woke at 3am out of habit and reached for the phone. That is what a badly designed on-call does. It does not just cost you a few bad nights. It takes your most reliable, most senior, most loyal people — exactly the ones who answer the page, who do not let it drop, who feel responsible — and it burns them down to ash, one 3am page at a time.

I want to be very precise about the claim I am making in this post, because it is the whole thesis: **on-call is a system you design, and a bad one is a design defect, not a willpower problem.** When a three-person team is paging thirty times a week each and the best people are quitting, the answer is not "be more resilient" or "we just need people who can handle the pressure." The answer is that the rotation is understaffed, the alerts are noisy, and the nights are uncovered — three concrete, fixable engineering problems. You do not motivate your way out of a broken rotation any more than you motivate your way out of a memory leak. You diagnose it, you measure it, and you fix the design. The figure below shows the two ends of that design space: the burnout rotation that loses its best people, and the humane one that the same workload can sustain for years.

![A two column before and after diagram contrasting a three person rotation paging thirty times a week per person that burns people out against a six person rotation with fixed alerts and follow the sun nights that drops the load to three pages a week and stops the attrition](/imgs/blogs/designing-a-humane-on-call-1.png)

By the end of this post you will be able to do five concrete things. You will be able to size a rotation from team size using the one-in-N math, so you know whether your rotation is sustainable before anyone burns out. You will be able to write an escalation policy that auto-escalates an unacknowledged page so no incident is ever silently dropped. You will be able to set a page budget — a hard ceiling on pages per shift — that turns "the on-call is tired" into a number you can act on, tied directly to your alerting. You will be able to run a handoff ritual and an onboarding ladder so nobody inherits a cold pager or carries it solo on day one. And you will be able to build an on-call health scorecard out of signals you already collect, so you find out the rotation is unsustainable from a dashboard and not from a resignation letter.

This is the operations layer of reliability. It sits inside the loop this whole series circles around — **define reliability, measure it, spend the error budget, reduce toil, respond to incidents, learn, and engineer the fix.** On-call is the human substrate underneath "respond to incidents." If you have not read the intro to the series, [reliability is a feature, not an accident](/blog/software-development/site-reliability-engineering/reliability-is-a-feature-the-sre-mindset) lays out the whole loop and why the error budget is the currency that ties it together. This post is about making the people who carry that loop's pager able to do it next year, and the year after that.

## 1. On-call is a rotation you size, not a fate you accept

Start with the most basic question and the one teams skip: how many people are on the rotation, and how often does each one carry the pager? Almost every on-call horror story traces back to a number that was never chosen on purpose. Someone stood up a service, three of the people who built it took turns carrying the pager because there were only three of them, and that ad-hoc arrangement quietly became permanent. Nobody ever sat down and asked: is one week in three a load a human can carry indefinitely? The answer is no, and the math says so before anyone gets hurt.

The core arithmetic is almost embarrassingly simple. If a rotation has $N$ people and each takes a turn as primary on a fixed cadence, then over the long run each person is primary a fraction $1/N$ of the time. With three people, that is one week in three. With six, one week in six. With eight, one week in eight. This single fraction — call it the **rotation frequency** — is the dominant input to whether on-call is humane, and it is set entirely by team size. You cannot fix an understaffed rotation with better tooling or more discipline. A one-in-three rotation means two-thirds of your weeks are spent either on-call or recovering from being on-call, and that does not leave enough clear time for a human to reset.

There is a widely cited rule of thumb from Google's SRE practice, and it is a good anchor: aim for roughly **six to eight people per rotation**. Six gives you one week in six as primary, which means five weeks clear out of every six — enough that on-call feels like an occasional duty rather than a permanent condition. Eight is more comfortable still. Below six it gets tight fast: a four-person rotation is one week in four, which is survivable but fragile, because the moment one person goes on vacation or leaves, you are back to one in three. Three or fewer is a rotation that will burn people out as a matter of arithmetic, no matter how good they are.

Why six to eight specifically, and not three, and not twelve? The lower bound is about recovery: a person needs enough clear weeks between shifts that the dread and the sleep debt fully reset, and one-in-six is roughly where that happens for a moderately noisy service. The upper bound is about competence: if you are on-call only one week in twelve, you carry the pager so rarely that you forget the runbooks, you have not seen the current architecture, and your first instinct in an incident is rusty. There is a real skill to running an incident, and skills decay without practice. So the band is bounded below by burnout and above by atrophy, and six to eight is the sweet spot where people are on often enough to stay sharp but rarely enough to stay sane.

There is a second, less obvious reason the lower bound bites so hard, and it is worth making explicit because it changes how you reason about adding people. The relationship between rotation size and clear time is *non-linear* in exactly the direction that hurts the small team. Going from three to four people raises your clear-week fraction from $2/3$ off-duty to $3/4$ off-duty — a real improvement, but not dramatic. Going from four to six raises it to $5/6$. But the *marginal* relief from each added engineer shrinks as the rotation grows: the fifth person you add to a four-person rotation buys far more sanity per head than the ninth person added to an eight-person rotation. This is why the band tops out around eight. Past that, each additional body buys you almost nothing in recovery time but costs you a meaningful slice of everyone's incident-handling practice, because the same number of real pages is now divided across more people who each see fewer of them. The function you are optimizing is not "maximize clear time" — it is "enough clear time to recover, with enough pager exposure to stay competent," and that joint constraint has a genuine interior optimum rather than a corner solution.

The other thing the one-over-$N$ math hides is *variance*. The fraction $1/N$ is the long-run average, but on-call load does not arrive smoothly. A quiet rotation can go three shifts with two pages total and then catch a brutal week where a bad deploy, a dependency outage, and a capacity event all land on the same unlucky primary. With three people, the unlucky primary draws that hand one week in three; with six, one week in six. So a bigger rotation does not just lower the average load — it lowers the probability that any one person draws the catastrophic week, and it gives the rotation slack to absorb that week when it comes (the secondary can take overflow, the off-duty engineers can be pulled in without anyone having just finished their own brutal shift). Sizing for the average alone is the classic mistake; you size for the bad week, and the bad week is what a thin rotation cannot survive.

The matrix below makes the trade-off concrete: rotation frequency is just team size in disguise, and the verdict column tells you whether each size is sustainable.

![A four row comparison matrix showing how team size sets rotation sustainability with three people on every third week burning out and six to eight people giving a one in six to one in eight cadence that is humane](/imgs/blogs/designing-a-humane-on-call-5.png)

#### Worked example: sizing a rotation for a six-person team

You have a six-engineer team owning one production service. You want a humane weekly rotation. Here is the design and the math behind it.

Each engineer is **primary one week in six**, which means each person carries the front-line pager about 17% of the year — roughly 8.7 weeks out of 52. You run a **secondary** alongside primary (more on why in the next section), and you stagger it so each person is also secondary one week in six, offset from their primary week. With the offset, a given engineer's calendar looks like this in a six-week cycle: one week primary, one week secondary, and **four weeks fully clear**. Over a year that is about 8.7 primary weeks, 8.7 secondary weeks, and roughly 35 weeks with no pager duty at all.

Is that sustainable? The clear-week ratio is the tell. Four clear weeks out of every six means two-thirds of the time a person has zero pager obligation, and the secondary week is far lighter than primary — secondary only gets paged when primary misses, which on a healthy rotation is rare. So the felt load is close to one heavy week and one light week in six. That is a load a person can carry for years. Compare it to the three-person team's two-out-of-three weeks on duty and you can feel the difference in your chest.

Now run the same arithmetic for the team you are tempted to staff with. Below is the relief curve laid out as a table, so you can see exactly what each headcount buys. The "felt load" column is the one that matters: it is the fraction of the year a person is either carrying primary or recovering from it, which I model as the primary fraction plus a small allowance for the secondary week (lighter, but not free).

| Rotation size | Primary frequency | Clear weeks per cycle | Primary weeks/year | Verdict |
| --- | --- | --- | --- | --- |
| 2 people | 1 in 2 | 0 of 2 | ~26 | Unsustainable — no recovery, no slack for a single absence |
| 3 people | 1 in 3 | 1 of 3 | ~17 | Burns people out; one departure puts you at 1-in-2 |
| 4 people | 1 in 4 | 2 of 4 | ~13 | Survivable but fragile; treat as a hiring case, not steady state |
| 6 people | 1 in 6 | 4 of 6 | ~8.7 | Humane; the recommended floor for off-hours paging |
| 8 people | 1 in 8 | 6 of 8 | ~6.5 | Comfortable; the top of the band before competence decays |
| 12 people | 1 in 12 | 10 of 12 | ~4.3 | Too sparse; people forget the runbooks between shifts |

Notice how the verdict flips between three and six and then stops improving meaningfully. The whole argument for "six to eight" is visible in two columns of that table: clear weeks climb fast up to six, then the marginal weeks bought get cheap while the per-person pager practice gets thin.

The grid below is the actual schedule for that six-person team. Read it as a calendar: each column is a week, the top two rows show who is primary and who is secondary, and the bottom rows show that every engineer gets four clear weeks out of six.

![A seven by seven schedule grid for a six person weekly rotation showing each engineer as primary one week in six and secondary one week in six with four fully clear weeks out of every six](/imgs/blogs/designing-a-humane-on-call-2.png)

### Rotation length: week, day, or split?

The other rotation lever is shift length. The two common choices are weekly and daily, and they trade off context against fatigue.

A **weekly rotation** — you are primary for seven days straight — is the default for most teams, and for good reason. The on-call engineer builds up context over the week: they know which deploy went out Tuesday, they remember the flaky alert from Wednesday night, they have the current state of the world in their head. Handoffs happen only once a week, so there is less ritual overhead. The cost is concentration: a bad week with several night pages lands entirely on one person, and there is no relief until Sunday.

A **daily rotation** spreads the pain — no single person gets a whole bad week — but it shatters context. The person on Tuesday has no memory of Monday's near-miss, every day starts cold, and you pay the handoff tax seven times instead of once. Daily rotations make sense for extremely high-volume rotations where a full week would be brutal, or for follow-the-sun setups where the "shift" is naturally a workday in one region.

A common refinement is the **weekday/weekend split**: one rotation owns Monday-through-Friday and a separate, often shorter rotation owns the weekend, so the weekend burden is shared out rather than always falling on whoever happens to be primary that week. This matters because weekend pages are disproportionately painful — they eat the time people most need to recover — and a split keeps any one person from drawing a weekend more than a fair share of the time.

The table below lays the three shapes side by side so you can match the shift length to your service rather than defaulting to whatever the last team did.

| Shift shape | Context retained | Fatigue concentration | Handoff cost | Best for |
| --- | --- | --- | --- | --- |
| Weekly | High — one head holds the week's state | High — a bad week lands on one person | Low — once a week | Most teams; moderate, bursty page volume |
| Daily | Low — every day starts cold | Low — no one draws a whole bad week | High — paid 7x per week | Very high steady volume; follow-the-sun shifts |
| Weekly + weekend split | High on weekdays | Medium — weekends shared separately | Medium — weekday and weekend handoffs | Teams with painful weekend pages |

My default recommendation: **weekly primary, with a weekend that rotates independently of the weekday primary** if your weekend page volume is non-trivial. Weekly gives you context; the weekend split keeps the recovery time fair. Daily only when the volume genuinely demands it, and if it does, that volume is itself a bug — see the page budget section.

One more refinement worth naming, because it is the cheapest fairness fix available: **avoid the back-to-back handoff trap.** If your weekly rotation runs Monday-to-Monday and your release train ships every Monday morning, then the same poor primary inherits both the handoff and the riskiest deploy of the week in the same twelve hours. Offset the rotation boundary from your highest-risk recurring event — hand the pager over on a Wednesday, say, so the incoming on-call has a day or two of calm to absorb context before the deploy storm hits. It is a one-line change to the schedule generator and it removes a predictable, recurring bad day from every shift.

## 2. The escalation chain: a page must never die quietly

Here is the failure mode that keeps me up at night more than any noisy alert: the page that fired, found nobody, and went silent. The primary's phone was on do-not-disturb. Or it was charging in the kitchen. Or they had food poisoning and could not get up. Or — and this is the one nobody admits — they had simply learned to sleep through the pager because it had cried wolf so many times. Whatever the reason, the page went out, nobody acknowledged it, and the system, having no further instructions, did nothing. The incident burned for forty minutes before a customer complained loudly enough to reach someone awake.

The fix is an **escalation chain**: an explicit, automated sequence of who gets paged, in what order, on what timer, so that an unacknowledged page climbs to the next person instead of dying. The principle is that *acknowledgment is the only thing that stops the climb.* A page is not "handled" because it was sent; it is handled because a human pressed the button that says "I have this." Until that happens, the page is the system's problem, and the system must keep escalating.

The standard shape is **primary, then secondary, then the on-call manager or incident commander.** Primary gets paged immediately and has a short window — five minutes is typical — to acknowledge. If they do not ack in that window, the page automatically escalates to the secondary, who gets another short window. If the secondary also does not ack, it escalates again, to a manager or a designated incident commander whose job at that point is not to fix the bug but to *get a human on it* — to pull people into a bridge, to wake whoever is needed, to make sure the incident is owned. That final rung is the "nobody answered" path, and every escalation policy needs one, because the whole point is that there is no terminal state where a page is unowned and the chain has given up.

The diagram below traces a single page through that chain. Note the structure: it branches at every rung, because at each level the page can either be acknowledged (the chain stops, success) or time out (the chain climbs). Only an acknowledgment is a terminal success; a timeout is never terminal.

![An escalation chain diagram showing a page climbing from primary at five minutes to secondary at ten minutes to the on call manager so no incident is silently dropped because one person was asleep](/imgs/blogs/designing-a-humane-on-call-3.png)

### The artifact: an escalation policy you can configure

Here is what that chain looks like as a real escalation-policy config, in the YAML-ish shape that PagerDuty, Opsgenie, and Grafana OnCall all model. The exact keys differ by vendor, but the structure is universal: an ordered list of rules, each with a set of targets and a timeout after which it escalates to the next rule.

```yaml
escalation_policy:
  name: "payments-api on-call"
  # An unacknowledged page climbs this ladder. Only an ack stops the climb.
  rules:
    - level: 1
      targets:
        - schedule: payments-primary      # whoever is primary this week
      escalate_after: 5m                  # no ack in 5 min -> go to level 2

    - level: 2
      targets:
        - schedule: payments-secondary    # the secondary on the same rotation
      escalate_after: 5m                  # still no ack -> go to level 3

    - level: 3
      targets:
        - schedule: payments-oncall-manager
        - user: incident-commander-rota
      escalate_after: 10m                 # the nobody-answered path

    - level: 4
      # last resort: page the whole team. Loud on purpose.
      targets:
        - team: payments-engineering
      # no escalate_after: this is the floor, the chain cannot fail past here

  # If two pages for the same service fire within this window, group them
  # so the on-call gets one notification, not a storm.
  alert_grouping:
    type: intelligent
    window: 5m
```

A few design choices in that config are worth calling out. The level-3 rung names both an on-call manager schedule and an incident-commander rotation, because the "nobody answered" situation is exactly when you want more than one phone ringing. The level-4 rung pages the entire team and has no timeout — it is the floor, the guarantee that there is no way for a page to escalate off the bottom of the policy and vanish. And the `alert_grouping` block at the end is doing quiet but important work: during a real incident, a dozen related alerts can fire in seconds, and without grouping the on-call gets a dozen separate pages for one problem. Grouping turns that storm into a single notification with a count. (For the deeper treatment of why grouping and inhibition matter, see the sibling post on [alerting that doesn't cry wolf](/blog/software-development/site-reliability-engineering/alerting-that-doesnt-cry-wolf).)

#### Tuning the escalation timers

The `escalate_after` values are not arbitrary, and getting them wrong fails in both directions. Set them too long and the chain is too slow: a five-minute primary window plus a five-minute secondary window plus a ten-minute manager window means twenty minutes can elapse before a guaranteed human is engaged, which is fine for a service with a generous error budget and a tail of recoverable degradations, but unacceptable for a tier-0 service where every minute of downtime is measured in real money. Set them too short and you manufacture noise: a two-minute primary window will escalate to the secondary every time the primary is in the bathroom or driving, doubling the page volume for no reliability gain and teaching the secondary to ignore "escalation" pages because most of them turn out to be false.

The way I tune them is to anchor on the service's severity and the human reality of how fast someone can actually get to a laptop. A reasonable matrix: for a tier-0 customer-facing service, primary window of three minutes and secondary of three, so a human is guaranteed within about six minutes; for a tier-1 service, five and five; for a tier-2 internal service, ten minutes at each rung, because a slightly slower response is an acceptable trade for not waking two people over something minor. The number you should never set is *zero* simultaneous paging across two rungs as your default — paging primary and secondary at the same instant for every alert feels safe but trains both of them to assume the other will take it, which is the bystander effect wired straight into your pager.

There is also a subtlety in *what counts as acknowledgment*. Acknowledging a page should mean "I am awake, I have seen this, and I am taking it" — not "I tapped the notification to make my phone stop buzzing." If your tooling lets a page be acked from a lock screen without opening the incident, you will get phantom acks: the primary swats the notification half-asleep, the chain stops climbing, and nobody is actually working the incident. The fix is to require that an ack be paired with engagement — joining the incident channel, claiming the incident in the tool — and to add a *re-page if no engagement* rule that resumes the climb if an acked incident shows no activity within a few minutes. The acknowledgment must mean something, or the whole chain is theater.

### Stress-testing the chain

A design is only as good as its behavior under the cases nobody planned for. Let me stress-test this one.

**What if the primary is asleep and never acks?** The page escalates to secondary at 5 minutes, then to the manager at 10. The cost is a 10-minute delay before a guaranteed human is engaged — not zero, but bounded and known, which is the whole point. You can tighten the timers for higher-severity services.

**What if two incidents overlap?** The primary acks the first and is now heads-down. The second page fires, the primary does not ack it because they are busy, and it escalates to secondary at 5 minutes. This is exactly the behavior you want: the secondary exists precisely to absorb the overflow when the primary is saturated. A single-on-call rotation with no secondary has no answer to overlapping incidents, which is one of the strongest arguments for always running a secondary.

**What if the secondary is also paged out?** Then it climbs to the manager, who pulls in additional responders. If your rotation regularly saturates both primary and secondary at once, that is a capacity signal: you are running too hot, and the page budget (next section) will already be screaming about it.

**What if the alerting backend itself is down and no page fires at all?** This is the scariest case, and the escalation policy cannot help you here — you cannot escalate a page that never fired. The defense is a **dead-man's switch**: a heartbeat alert that fires if your monitoring stops reporting, watched by an independent path. It is the one alert you wire to page even when everything else is silent, because silence is itself a symptom.

## 3. Follow-the-sun versus covering your own nights

The single cruelest thing about on-call is the 3am page. Not because the work is hard at 3am, but because being woken from deep sleep wrecks you for the next day in a way that is hard to overstate — the research on sleep fragmentation is bleak, and anyone who has been on-call knows the next-day fog in their bones. A rotation that regularly wakes people at 3am is not sustainable no matter how few pages there are, because the *interruption* is the damage, not the workload.

There are two ways out of this, and which one you can use depends entirely on whether your organization spans timezones.

If you have engineers in multiple regions — say, Americas, Europe, and Asia — you can run a **follow-the-sun** rotation. The pager follows daylight around the globe. The Americas team holds it during their working hours, then hands it to Europe as their day ends, then Europe hands it to Asia, then Asia back to the Americas. Every shift is worked in someone's daylight. Nobody is ever woken at 3am, because at 3am in any given region, the pager is being held by a team that is wide awake on the other side of the planet. This is the gold standard for night coverage, and it is the single biggest quality-of-life improvement available to a global engineering org.

Follow-the-sun is not free, though, and the place it goes wrong is the handoff at the region boundary. When the Americas team's day ends and Europe picks up the pager, every piece of context that does not transfer becomes a 9am surprise for a fresh team that has no memory of the incident that was simmering when the Americas team logged off. The cure is the same handoff ritual covered in section 6, but applied at *every* region boundary rather than once a week — which means follow-the-sun pays the handoff tax two or three times a day instead of once a week. That is the real cost of the model, and it is why a follow-the-sun rotation only works when the handoff is disciplined: a written, structured pass of open incidents, fragile systems, and active silences at each boundary, not a verbal "all quiet" in a chat thread. The reward is worth the discipline — zero night pages is transformative — but you do not get the reward without paying the handoff cost honestly.

There is also a coverage-gap trap to avoid. The three regions almost never have working hours that tile the 24-hour clock with no seams, so you get overlap windows (two regions both holding the pager, which is fine, even good) and risk gaps (a stretch where one region has logged off and the next has not quite started). Map your regions' actual working hours against the clock before you commit to follow-the-sun, and if there is a seam, close it explicitly — extend one region's shift by an hour, or designate a thin overlap rather than discovering at 6am that the pager spent ninety minutes owned by nobody.

If you are a single-region team, follow-the-sun is not on the table, and you have to cover your own nights. The honest options are worse: either someone is on-call overnight and may get woken (the default, and the one that burns people out), or you accept a *degraded nighttime SLO* and explicitly do not page for anything below a certain severity at night — a 3am page only for a full outage, with everything else waiting until morning. That second option is more humane than it sounds and is underused. If your service genuinely does not need a human touching it at 3am for a partial degradation, then do not wake a human at 3am for a partial degradation. Tie the nighttime paging threshold to severity and let the rest queue. This is, again, the error budget thinking applied to humans: you spend a little reliability at night to buy your people sleep, and you do it on purpose with a documented threshold rather than by accident.

The way you implement that nighttime threshold is a time-based route in Alertmanager, and it is worth showing because it is the concrete mechanism behind "let the minor stuff wait until morning." You define a time interval for off-hours and route low-severity alerts to a non-paging receiver during that window:

```yaml
# Alertmanager: page only for Sev1 overnight; queue everything else for morning.
time_intervals:
  - name: overnight
    time_intervals:
      - times:
          - start_time: "22:00"
            end_time: "07:00"
        location: "America/New_York"

route:
  receiver: oncall-pager
  routes:
    # Overnight, anything below Sev1 goes to a backlog, not the pager.
    - matchers:
        - severity =~ "warning|info"
      active_time_intervals:
        - overnight
      receiver: morning-queue        # a non-paging receiver; reviewed at 09:00
    # Sev1 always pages, day or night. This route has no time gate.
    - matchers:
        - severity = "critical"
      receiver: oncall-pager
```

The discipline this encodes is that the *severity label is a promise about whether something is worth a human's sleep.* If everything is tagged critical, the route does nothing, so this artifact only works in concert with honest severity classification — which is a forcing function in itself, because it makes the team decide, for each alert, "is this worth waking someone for?" That question, asked once per alert at design time, is most of the battle.

The before-and-after below contrasts the two worlds: the single-region rotation where nights are unavoidable, and the follow-the-sun handoff where every shift is in daylight.

![A two column before and after diagram contrasting a single region rotation where a 3am page disrupts sleep and causes next day fog against a follow the sun handoff across three regions where every shift is worked in daylight with zero night pages](/imgs/blogs/designing-a-humane-on-call-4.png)

#### Worked example: what a 3am page actually costs

Let me put a number on the night page, because "it's bad" is not a budget. Suppose a single-region rotation pages, on average, twice a week at night. Each night page wakes the engineer, costs maybe 20 minutes of acknowledged active work, and — this is the expensive part — degrades the next full day. Sleep research suggests a fragmented night can cut next-day effective output by something like 20 to 30 percent; call it a quarter of a workday lost, conservatively.

Two night pages a week, each costing a quarter-day of next-day productivity, is half a workday a week of lost output from one person — call it roughly 25 lost workdays a year on a continuous rotation, and that is *before* you count the slow accumulation of sleep debt that pushes good engineers toward the exit. (These figures are illustrative, not measured from a specific company; the orders of magnitude are what matter.) Set against that, the cost of standing up a follow-the-sun handoff — or of explicitly not paging below Sev1 at night — is almost always the cheaper trade. The night page is not free just because nobody invoices you for it.

## 4. The page budget: an unsustainable shift should set off an alarm

Here is the lever that ties on-call health directly back to the rest of the SRE loop, and it is the one most teams are missing: a **page budget.** Just as an error budget puts a hard number on how much unreliability you will tolerate, a page budget puts a hard number on how many pages a single on-call shift can absorb before the rotation is, by definition, unsustainable. My rule of thumb, again anchored in Google's SRE practice, is a ceiling of about **two incidents per on-call shift.** Not two pages — two *incidents*, where an incident may bundle several related pages. More than two genuine incidents in a shift means the on-call cannot give each one proper attention, and the shift has tipped from "handling problems" to "firefighting."

The reasoning behind two is the same reasoning behind any budget: it forces a conversation that would otherwise never happen. If a shift regularly exceeds two incidents, exactly one of three things is true, and all three are bugs you can fix. Either the rotation is **understaffed** (too few people absorbing the real incident load, fix by hiring or by splitting the service), or the alerts are **too noisy** (most "incidents" are false pages that should never have fired, fix the alerts), or the service is **genuinely too unreliable** (it is breaking more than two real times a shift, fix the service by spending engineering time on the top failure modes). The page budget does not tell you which of the three it is, but it makes the unsustainability *visible and undeniable*, which is the hard part. Without it, "the on-call has been rough lately" is a feeling that gets dismissed; with it, "we are running at 3.5 incidents per shift against a budget of 2" is a number that demands a decision.

The crucial design principle here, and I want to say it plainly: **a noisy rotation is a bug to fix, not a tax to pay.** The single most damaging belief in operations culture is that pages are just the cost of running a service and the on-call's job is to endure them. No. Every page that did not require a human to act *right now* is a defect — in the alert, in the service, or in the rotation sizing — and it should be filed, triaged, and fixed exactly like any other bug. The page budget is the mechanism that converts that belief into practice, because it gives you a threshold that, when breached, generates a ticket.

### The artifact: tracking pages per shift in Prometheus

You can measure this with the alerting you already have. If your alerting pipeline emits a metric on every page — and Alertmanager does, you can scrape `alertmanager_notifications_total` — you can record pages per on-call shift and even alert on the page budget itself. Here is a recording rule that computes pages per on-call-week per person, plus a meta-alert that fires when the rotation breaches its page budget.

```yaml
groups:
  - name: oncall-health
    interval: 1h
    rules:
      # Pages delivered per week, per receiver (the on-call rotation).
      - record: oncall:pages_per_week
        expr: |
          sum by (receiver) (
            increase(alertmanager_notifications_total{
              integration="pagerduty", state="active"
            }[7d])
          )

      # Off-hours pages: the ones that wake people. Tag-based, see note.
      - record: oncall:offhours_pages_per_week
        expr: |
          sum by (receiver) (
            increase(alertmanager_notifications_total{
              integration="pagerduty", state="active", hours="offhours"
            }[7d])
          )

  - name: oncall-budget-alerts
    rules:
      # The page budget alert: more than ~14 pages/week (~2/shift over
      # a 7-day week) means the rotation is running unsustainably hot.
      - alert: OnCallPageBudgetExceeded
        expr: oncall:pages_per_week > 14
        for: 0m
        labels:
          severity: ticket          # routes to a backlog, NOT to the pager
        annotations:
          summary: "On-call page budget exceeded for {{ $labels.receiver }}"
          description: >
            {{ $labels.receiver }} received {{ $value | humanize }} pages in
            the last 7 days against a budget of 14 (~2/shift). The rotation is
            understaffed, the alerts are noisy, or the service is too
            unreliable. File a remediation, do not just endure it.
```

Two details matter. First, the severity on the meta-alert is `ticket`, not `page` — the alert about too many pages must never itself be a page, or you have made the problem worse. It routes to a backlog where someone triages it during working hours. Second, the off-hours recording rule depends on your alerts being tagged with whether they fired during business hours; you can do this in Alertmanager with a time-based routing label, and it is worth the effort, because off-hours pages are the ones that actually hurt.

#### Worked example: a page budget that flags an unsustainable shift

Take the six-person rotation from earlier. Each weekly shift has, say, 5 working days and 2 weekend days, and you have set a page budget of 2 incidents per day equivalent — but pragmatically you track it as pages per week, so the weekly budget is roughly 14 pages (2 per day for 7 days is generous; many teams set the weekly ceiling lower, around 10). The `OnCallPageBudgetExceeded` alert above fires at 14.

Now suppose the dashboard shows the rotation averaging 22 pages a week. That is 8 over budget, every week, landing on one person each shift. The page budget does its job: it generates a ticket that says, in effect, "this rotation is running at 157% of its sustainable load." The next move is diagnosis — pull the pages, classify them, and find out which of the three causes is dominant. In nearly every case I have seen, the answer is *noise*: a handful of badly written alerts account for the majority of the volume, and silencing or fixing them drops the rate below budget without hiring anyone. But you only go looking because the budget told you to.

The classification step is worth doing properly, because the three causes have completely different fixes and you do not want to grow the rotation when the real problem is a single misconfigured threshold. The discriminating question for every page is: **did a human have to do something, right now, that could not have waited?** Sort the 22 pages into three buckets and the fix declares itself.

| Bucket | What it looks like | Typical share | The fix |
| --- | --- | --- | --- |
| Noise | Self-recovered before anyone acted; duplicate of one root cause; cause-based alert that never hurt a user | 60–80% | Fix or delete the alert; group duplicates; convert to symptom-based |
| Real but deferrable | A genuine problem, but it could have waited until morning | 10–25% | Re-route off the pager to a daytime queue; lower its severity |
| Real and urgent | A user is being hurt now and a human had to act | 10–20% | This is the true load — size the rotation and harden the service for this |

The arithmetic of that table is the whole diagnosis. If 22 pages a week sort into roughly 15 noise, 4 deferrable, and 3 urgent, then the *sustainable* load is 3 pages a week — comfortably under budget — and the rotation was never understaffed at all. It was drowning in 15 pages a week of pure noise plus 4 that should never have woken anyone. Fix the alerts and the rotation that looked 157% overloaded is suddenly 21% utilized. This is why the order of operations matters so much, and why "just add people" is so often the expensive wrong answer: you would have hired into a problem that a few hours of alert surgery dissolves entirely.

## 5. Compensation, fairness, and the volunteer-hero anti-pattern

Now the part that engineering-minded teams are weirdly reluctant to talk about: on-call is *work*, often done outside working hours, and work outside working hours should be compensated. This is not a soft HR point; it is a reliability point. An uncompensated, unfair on-call is an unstable system, because the people carrying it will, rationally, find ways to carry it less — by being slow to ack, by quietly opting out, or by leaving. Compensation and fairness are not perks bolted onto a good rotation; they are load-bearing parts of one.

There are a few honest models for compensation, and the right one depends on your jurisdiction and company, but the principle is constant: **the burden of off-hours availability is recognized.** Some companies pay an on-call stipend — a flat amount per on-call week, regardless of whether you got paged. Some pay per-page or per-incident on top of that. Some, especially where labor law requires it, give **time in lieu**: an hour of off-hours incident work buys back an hour (or more) of regular time off. The flat stipend is my preferred default because it compensates the *availability* — the constraint on your life of having to stay near a laptop and sober and reachable — and not just the pages, since the constraint exists whether or not the pager fires. A quiet on-call week is still a week you could not go hiking out of cell range.

The reason the choice of model matters beyond fairness is *incentives* — compensation quietly shapes behavior, and a badly chosen model can fight your reliability goals. The table below lays out the common models against what they reward and the trap each one carries.

| Model | What it pays for | The incentive it creates | The trap |
| --- | --- | --- | --- |
| Flat stipend per week | The constraint of being available | Neutral — paid the same whether quiet or busy | Costs the same even when the rotation is broken; pair with a page budget |
| Per-page / per-incident | Each actual interruption | Rewards a noisy rotation financially | Can make people quietly *prefer* noise; never use this alone |
| Time in lieu | Off-hours hours worked | Restores recovery time directly | Hard to take the time back on a busy team; can become an IOU never cashed |
| Stipend + time in lieu | Availability *and* heavy nights | Recognizes both the constraint and the cost | More bookkeeping; usually worth it |

The per-incident-only model is the dangerous one and the reason deserves stating plainly: if an engineer is paid meaningfully more for a busy on-call week than a quiet one, you have created a financial incentive *not* to fix the noise — the very noise that the page budget is trying to drive to zero. I have seen rotations where the unspoken understanding was that on-call was a nice earner, and unsurprisingly nobody was in a hurry to silence the alerts. Compensate the availability with a flat stipend so the incentive points the right way, and add time in lieu for genuinely heavy nights so the people who draw the bad week are made whole, but never make the paycheck grow with the page count.

Fairness is the second pillar, and it has three parts. The **schedule must be fair** — built well in advance, visible to everyone, with the load (especially weekends and holidays) spread evenly, and the grid from section 1 is exactly this. There must be a real **opt-out and swap mechanism** — life happens, people have weddings and surgeries and sick kids, and a rotation with no humane way to trade shifts forces people to either martyr themselves or feel guilty, both of which corrode the team. And you must actively fight the **volunteer-hero anti-pattern**.

The volunteer hero is the engineer who is so good and so responsible that they quietly take more than their share — they answer pages outside their shift, they jump on every incident, they let others swap out and never swap themselves, they are "always around." It feels great for everyone except them, and it is poison for the rotation for two reasons. First, it burns out your best person fastest, and you cannot afford to lose them. Second, and more insidiously, it *masks the system's real load*. When a hero is silently absorbing the overflow, the page budget looks fine, the rotation looks sustainable, and nobody fixes the underlying noise — until the hero leaves and the whole thing collapses, revealing that it was never sustainable, just propped up by one person's unpaid heroism. **The pager is a team responsibility, evenly distributed, and a hero quietly carrying extra is a measurement failure as much as a kindness.** The job of a good on-call design is to make the hero unnecessary, which means it must make the real load visible — which loops straight back to the page budget.

## 6. The handoff ritual: the baton must not drop

Every week (or every day, or every region boundary), the pager changes hands, and that moment is more dangerous than it looks. The outgoing on-call has a head full of context — the deploy that went out Friday afternoon, the cache tier that has been flaky, the alert they silenced until Monday, the incident that is mitigated but not yet resolved — and if none of that transfers, the incoming on-call starts cold. They get paged at 11pm for a symptom of Friday's deploy and spend an hour rediscovering what the outgoing person already knew. The handoff is where context goes to die, unless you make it a ritual.

A good handoff is a short, structured conversation — five to ten minutes, synchronous if possible, with a written record — that walks a fixed checklist. It is not a Slack thumbs-up emoji. The checklist below is the one I use, and the timeline figure shows it as a baton pass.

![A timeline diagram showing the shift handoff ritual covering open incidents fragile systems recent deploys active silences and a live pager test before the baton is passed to the incoming on call](/imgs/blogs/designing-a-humane-on-call-8.png)

### The artifact: a handoff checklist

```yaml
# On-call handoff checklist. Run at the shift boundary, outgoing -> incoming.
# Post the filled-out version in the team channel as the written record.
handoff:
  open_incidents:
    # Anything still live or mitigated-but-not-resolved.
    - id: INC-4471
      state: mitigated
      note: "DB failover done at 14:00; root cause still open, watch replica lag"
  fragile_systems:
    # Things that are healthy now but you'd watch closely.
    - "Cache tier flapped twice this week under load; runbook RB-12 if it pages"
  recent_deploys:
    # Deploys still inside their bake window — most likely to page.
    - "payments-api v3.4.1 shipped Fri 16:00, 3 services, watch error rate"
  active_silences:
    # Silenced alerts the incoming on-call must know are muted.
    - alert: DiskFillingSlowly
      expires: "Mon 09:00"
      reason: "known slow leak, ticket OPS-882, do not re-silence past expiry"
  scheduled_maintenance:
    - "Vendor DB upgrade window Sat 02:00-04:00, expect failover blips"
  pager_test:
    # The incoming on-call sends themselves a test page and confirms it arrives.
    confirmed: true
  acknowledged_by: "incoming-oncall@team"
```

Walk through why each line earns its place. **Open incidents** are obvious — the incoming person must know what is still burning. **Fragile systems** transfers the soft knowledge that does not show up in any dashboard: the thing that has been *almost* breaking. **Recent deploys** matters because the single best predictor of the next incident is the last deploy; anything still inside its bake window is a prime suspect, and the incoming on-call should know to look there first. **Active silences** is the one teams forget, and it is dangerous — a silenced alert is an invisible blind spot, and if the incoming on-call does not know it is muted, they will not understand why a real problem is not paging. **The pager test** is the smallest item and the one that has saved me the most grief: the incoming on-call sends themselves a test page and confirms it physically arrives on their phone, because a handoff to a pager that does not actually ring is no handoff at all.

## 7. Onboarding: nobody carries the pager solo on day one

The fastest way to break a new engineer is to hand them the pager in their second week, alone, at night, with a runbook they have never read for a system they have not seen break. They will either freeze in the incident — which is terrifying and humiliating and a great way to make someone quit — or they will make a panicked change that makes things worse. Both outcomes are the rotation's fault, not theirs. On-call competence is a skill, and skills are built in stages with support, not granted by adding someone to a schedule.

The onboarding ladder I use has four rungs, and a new engineer climbs them over roughly four to five weeks before they ever carry the pager unassisted. The tree figure shows the full ladder.

![A tree diagram showing the staged onboarding of a new on call engineer from learning the system through shadowing and a runbook tour to a reverse shadow and a supported first solo shift before full rotation](/imgs/blogs/designing-a-humane-on-call-7.png)

The first rung is **shadowing**: the new engineer is added as a silent observer to a veteran's shift. When a page fires, the veteran drives and the new person watches — they see the real workflow, how an experienced on-call triages, where they look first, how they decide whether to escalate. There is no pressure, because the new person's actions do not matter yet; they are learning the muscle memory by watching it.

The second rung is the **runbook tour**: a guided walk through the top ten most-likely pages and their runbooks. Not "here is the wiki, good luck" — an actual sit-down where a veteran walks the new person through the most common alerts, what each one means, and what the first three steps are. The goal is that when one of those top-ten pages fires for real, the new person has *seen it before* in a calm moment.

The third rung is the **reverse shadow**, and it is the most important and most often skipped. Now the new engineer drives — they hold the pager, they triage, they make the calls — and the veteran watches silently, intervening only if something is about to go badly wrong. This is the flight-instructor model: hands on the controls, instructor's hands hovering nearby. It is where confidence actually forms, because the new person learns they can do it while a safety net is right there.

The fourth rung is the **supported first solo shift**: the new engineer is now primary, on their own, but with a named buddy explicitly on standby — not in the escalation chain, just a person they have been told "text me anytime, no question is too dumb, I'd rather you ask." Only after a clean supported solo shift do they roll into the full one-in-six rotation like everyone else. The whole ladder is, again, a system: it converts "throw them in and hope" into a repeatable process that produces a competent, confident on-call every time.

Two practical notes make the ladder hold up in the real world. First, the right time to climb it is not the calendar but *exposure*: a four-to-five-week window is the floor, but the actual gate is whether the new engineer has handled — under supervision — a representative spread of the top pages. If their shadow and reverse-shadow weeks happened to be quiet, they have served the time but not built the muscle, and you extend rather than promote them on schedule. Onboarding readiness is a competence check, not a tenure check. Second, the ladder needs a deliberate place for the new engineer to *fail safely*, and the best one is a game day. Rather than waiting for production to break during their reverse shadow, you inject a controlled fault in a staging environment — kill a dependency, fill a disk, trip a circuit breaker — and let them run the incident end to end with the veteran watching. A new on-call who has already navigated a simulated database failover is dramatically calmer when the real one fires at 3am, because the first time they touch the runbook is not the night it counts. (Chaos and game days are their own discipline; for the systematic version, the resilience-testing post in this series goes deep, and the architecture-level fault-injection patterns live in the system-design track.)

The onboarding ladder also pays a dividend most teams miss: it is the best documentation audit you will ever run. Every time a new engineer climbs the runbook tour and asks "wait, what does this alert actually mean?" or hits a runbook step that no longer matches reality, they have found a defect in your operational docs that the veterans had stopped seeing because they had memorized around it. Treat every such question as a bug report against the runbook, fix it while the confusion is fresh, and your onboarding process becomes a self-cleaning mechanism that keeps the runbooks honest. A rotation that onboards someone new every few months never lets its documentation rot, because rot gets caught on contact.

## 8. Psychological safety: it is OK to escalate, OK to not know

Every technical control in this post fails if the culture around the pager is wrong, and the cultural failure mode is shame. The engineer who feels they will look incompetent if they escalate will sit on an incident too long trying to solve it alone. The engineer who feels they should already know everything will not ask the question that would have resolved the incident in two minutes. The engineer who is afraid of blame will hide the contributing factor in the postmortem. Every one of these is the rotation working against itself, and the antidote is psychological safety — built deliberately, stated explicitly, and modeled by the senior people.

There are three norms I state out loud to every on-call, repeatedly, because they have to be heard more than once to be believed. **It is OK to escalate.** Escalating is not failure; it is the system working as designed. The escalation chain exists *to be used*, and an on-call who pulls in the secondary or the manager because they are over their head has made exactly the right call. The wrong call is the lonely hour spent stuck because asking felt like admitting weakness. **It is OK to not know.** Nobody holds the entire system in their head, the runbooks exist precisely because memory is unreliable, and "I don't know, let me get the person who does" is a senior move, not a junior one. **The pager is a team responsibility.** The person holding it this week is not solely responsible for every outcome; they are the team's representative on the front line, and the team owns the reliability of the service together. When an incident goes badly, the question is never "why did the on-call fail" — it is "what about our system let one tired person at 3am be the only thing standing between us and an outage."

The deepest version of this is blameless culture, which is a big enough topic that the postmortem post in this series treats it properly — when [the anatomy of an incident](/blog/software-development/site-reliability-engineering/the-anatomy-of-an-incident) ships it will go into how blameless review surfaces more truth, because people stop hiding the contributing factors once they trust they will not be punished for them. For on-call specifically, the load-bearing point is this: the senior people set the temperature. If the staff engineer escalates without embarrassment, asks "dumb" questions in the channel, and says "I got this wrong" in the retro, the juniors learn it is safe. If the senior people perform omniscience, the juniors learn to hide, and the rotation gets quietly more dangerous even as it looks calm.

## 9. Measuring on-call health and actually acting on it

You cannot manage what you do not measure, and the great failure of on-call is that teams measure the *service* obsessively and the *humans carrying it* not at all. The whole point of this section is that on-call health is a measurable property of the system, made of signals you mostly already collect, and that the only thing that makes the measurement worth anything is acting on it. A scorecard you look at and shrug is worse than no scorecard, because it lets you tell yourself you are paying attention.

The scorecard I build has five signals, and the stack figure below shows them from the most operational at the top to the most human at the bottom.

![A layered stack diagram of the on call health scorecard from pages per shift down through off hours pages time to ack sleep disruption and an actionable rate to a quarterly satisfaction survey that is acted on](/imgs/blogs/designing-a-humane-on-call-6.png)

The top signal is **pages per shift**, measured against the page budget from section 4 — target under two incidents per shift. This is the headline number, the one that tells you instantly whether the rotation is running hot. Below it is **off-hours pages per week**, the count that actually wakes people, with a target near zero, because as we saw the night page is the expensive one. Next is **time to acknowledge** — the p90 latency from page-fired to page-acked, target under five minutes — which is a quiet but powerful signal: if time-to-ack is creeping up, your on-call is starting to ignore the pager, and that is the early warning of the trust collapse that killed the alert in the cry-wolf post. Below that is **sleep disruption**, the nights woken per shift, which is the one signal you usually have to gather by asking rather than from a metric, but it is the one that most directly predicts burnout. And at the bottom, the most human and the most important: a **quarterly on-call satisfaction survey**, three or four honest questions ("was your last on-call sustainable? did you feel supported? would you want to do it again next quarter?"), the results of which are *visibly acted on*. The whole stack is worthless if the survey comes back red and nothing changes; the act of fixing what the survey surfaces is what makes people answer it honestly the next time.

The discipline here is the same as everywhere in SRE: **any layer in the red is a bug in the rotation, not a tax to keep paying.** Pages-per-shift over budget? Diagnose noise versus understaffing versus unreliability and fix the dominant cause. Off-hours pages climbing? Consider follow-the-sun or a nighttime severity threshold. Time-to-ack creeping up? Your people are losing trust in the pager — go find out why. The scorecard does not fix anything by itself; it tells you precisely where to point your engineering time so that the fix is targeted and not a guess.

The single most useful PromQL on this whole dashboard is the time-to-acknowledge percentile, because it is the leading indicator — it moves before the survey turns red and before anyone quits. If your incident tool exports a histogram of ack latency, you read the p90 directly:

```promql
# p90 time-to-acknowledge over the trailing 30 days, per rotation.
histogram_quantile(
  0.90,
  sum by (le, receiver) (
    rate(oncall_ack_latency_seconds_bucket[30d])
  )
)
```

Watch the *trend*, not just the level. A p90 that holds steady at three minutes is a rotation that trusts its pager. A p90 that has drifted from two minutes to seven over a quarter is a rotation quietly learning to ignore the pager, and that drift will show up in the satisfaction survey one quarter later and in a resignation one quarter after that. The metric buys you two quarters of warning, which is exactly enough time to fix the noise before it costs you a person.

#### Worked example: reading a scorecard that is going quietly wrong

Here is a scorecard for a six-person rotation that, on the surface, looks fine — nobody has complained, the service's SLO is green — but is drifting toward trouble. Read the five signals together.

Pages per shift averages 1.6, comfortably under the budget of 2, so the headline number says "healthy." Off-hours pages are running at 3 a week against a target near zero — not alarming on its own, but not zero either. Time-to-ack p90 has crept from 2 minutes a quarter ago to 6 minutes now. Sleep disruption, gathered from the survey, is 1.5 nights woken per shift. And the latest quarterly survey scored "would you want to do this again next quarter?" at a tepid average, with two free-text comments mentioning the night pages specifically.

A team looking only at the headline number would declare victory: pages-per-shift is under budget, ship it. But read the signals as a *system* and the story is different. The off-hours pages are low in count but high in cost — 3 a week, 1.5 of them waking someone — and the rising time-to-ack is the tell that the rotation is starting to resent the pager even though the volume is technically fine. This is a rotation that is sustainable on the spreadsheet and souring in reality, and the fix is targeted: not "grow the rotation" (pages-per-shift is fine) but "kill the night pages" — push the three off-hours alerts through the overnight severity route from section 3 so only a true Sev1 wakes anyone. Do that, and the predicted effect is off-hours pages toward zero, time-to-ack recovering as trust returns, and the next survey turning from tepid to green. The scorecard caught a problem two quarters before it would have become a resignation, which is the entire point of measuring the humans and not just the service.

## 10. War story: the three-person team that paged thirty times a week

Let me tell you the full version of the story I opened with, because the fix is the whole post in one narrative, and the before-and-after numbers are the kind of proof this series insists on.

The service was a payments-adjacent API, important enough that it had to be reliable but small enough that it had been built and operated by three engineers. Those three carried the pager, one week in three, and by the time anyone looked closely the rotation was averaging **about thirty pages per person per week** during an on-call week — a page roughly every five or six hours, around the clock. The on-call could not work on anything else during their week; they were a full-time pager-answering machine. Two of the three were visibly burning out, and the third, the best of them, was the volunteer hero quietly absorbing the overflow on the other two's weeks, which had hidden how bad it really was. When she gave notice, the truth came out: the rotation had been unsustainable for a year and was being held together by one person's unpaid heroism.

We fixed it in three moves, in order, and measured each one.

**First, we attacked the noise.** We pulled three weeks of pages and classified every one: was it actionable (a human had to do something right now) or not? About **80% were not** — they were threshold alerts on things that self-recovered, duplicate pages for one underlying problem, and cause-based alerts that fired on conditions that never actually hurt a user. We rewrote the worst offenders as symptom-based, multi-window burn-rate alerts (the technique in the cry-wolf post), grouped related alerts in Alertmanager, and silenced the genuinely useless ones with tickets to fix the underlying causes. **Pages dropped from about thirty per person per week to about three.** That single change — fixing the alerts, not the people — recovered most of the sustainability, and it cost zero new headcount.

**Second, we grew the rotation.** Three people is unsustainable arithmetic regardless of noise; even three real pages a week, one week in three, is a thin rotation with no slack for vacations. We made the case to hire, framing it explicitly in the page-budget language — "we are running at 4x the sustainable load and we have already lost one engineer to it" — which is a far more persuasive argument to a budget-holder than "on-call is hard." We grew the team to **six**, which moved everyone to one week in six with five clear weeks between shifts. The grid from section 1 is, in fact, that team's new schedule.

**Third, we covered the nights.** Even at three pages a week, some landed at 3am, and the company had a small engineering presence in another timezone. We set up a partial follow-the-sun handoff for the overnight hours, so the worst of the night pages were caught by people who were awake. For the nights still covered locally, we set a severity threshold: below Sev1, the page waited until morning.

The before-and-after, measured over the following two quarters: **pages per person per week, ~30 → ~3. Off-hours pages, several a week → near zero. On-call satisfaction survey, a sea of red → mostly green. Attrition attributable to on-call, one senior engineer lost → zero.** And here is the part that should convince any skeptic: the *service* got more reliable too, because the engineers who used to spend their on-call week drowning in noise now had the slack to actually fix the top failure modes. A humane on-call is not in tension with a reliable service. It is a precondition for one, because the only people who can make a service reliable are the people who are not too burned out to try.

There is a sequencing lesson buried in those three moves that is easy to miss, so let me make it explicit, because getting the order wrong is the most common way this fix goes sideways. We attacked the noise *first*, before hiring and before touching the nights, and that order was deliberate. Had we hired first, we would have onboarded three new engineers into a rotation that was 80% noise — meaning we would have taught three new people that on-call is a miserable firehose, and the noise would have stayed exactly as loud, just spread thinner. Had we set up follow-the-sun first, we would have exported the noise to another timezone, exporting the burnout with it. Noise reduction had to come first because it is the only move that shrinks the actual problem rather than redistributing it. Hiring and night coverage are how you make a *real* load sustainable; they do almost nothing for a *fake* load made of bad alerts, and applying them to a fake load just hides it behind a healthier-looking schedule. Fix what is broken, then size for what remains.

It is also worth being precise about how we *measured* the before-and-after, because "pages dropped from thirty to three" is only trustworthy if the measurement is honest. We did not eyeball it. We had the `oncall:pages_per_week` recording rule from section 4 running before we changed anything, so the baseline of roughly thirty pages per person-week was a recorded number, not a memory. After each of the three moves we let the rotation run two full weeks and read the rule again, which is how we could attribute most of the drop specifically to the noise work rather than crediting the hire. And the human signals — off-hours pages, time-to-ack, the satisfaction survey — came from the scorecard in section 9, gathered the same way before and after so the comparison was apples to apples. The credibility of a reliability win lives entirely in whether you were measuring the same thing the same way on both sides of the change. If you only start measuring after you fix something, you have a story, not a result.

This is a realistic composite, not a single literally-true company case, but every number in it is the order of magnitude I have seen repeatedly, and the 80%-noise figure in particular is depressingly common — most overloaded rotations are overloaded by noise, not by real failures.

## 11. War story: the page that found nobody

A shorter one, on the escalation chain. A team I knew ran a single-on-call rotation — primary only, no secondary, no escalation past primary. It worked fine for two years, which is exactly the problem, because "it worked fine" let everyone forget it had no failure path. Then the primary one week was an engineer who, unknown to anyone, had started a new medication that knocked him out cold at night. A real incident fired at 1am. The page went to his phone. He did not wake up. There was no secondary to escalate to and no manager rung, so the page sat, unacknowledged, for **over an hour**, while a database ran out of connections and the service returned errors to a growing fraction of users. It was finally caught when an engineer in another timezone happened to notice the error-rate graph and raised the alarm manually.

The fix was the escalation chain in section 2 — a secondary, and a manager rung past that, and a dead-man's switch so that even total monitoring silence would page someone. None of it was exotic; all of it was the difference between a one-hour outage and a five-minute one. The lesson is the one that runs through this whole post: **the failure path is the design.** A rotation that only works when the primary is awake and reachable is not a rotation, it is a coin flip, and the coin had been landing heads for two years right up until it didn't. You do not get to find out your escalation chain is missing during the incident that needed it.

## 12. How to reach for this (and when not to)

Everything in this post has a cost, and the principal-engineer move is to know when the cost is not worth it. Here is the honest guidance.

**Run a secondary and an escalation chain for anything that pages a human off-hours.** This is close to non-negotiable. The cost is low — a secondary's week is light if the rotation is healthy — and the failure it prevents (a dropped page) is severe. The only services that can skip it are ones where a missed page genuinely does not matter, which means they probably should not be paging off-hours at all.

**Size for six to eight when you can, but do not let perfect block good.** If you only have four people, a one-in-four rotation with a ruthless page budget and aggressive noise reduction can be sustainable for a while — but treat it as a known risk and a hiring case, not a steady state. Do not pretend a three-person rotation is fine; it is borrowing against your best people, and the loan comes due.

**Do follow-the-sun only if you genuinely have the timezone coverage.** A pretend follow-the-sun, where one region is two people who are themselves overloaded, is worse than honest single-region night coverage, because it spreads the same insufficient capacity thinner. If you cannot cover a region's daytime properly, do not hand it the pager.

**Set a page budget always, even for a healthy rotation.** It is nearly free — a recording rule and a ticket-severity alert — and it is the early-warning system that catches degradation before it becomes attrition. There is no rotation too small or too healthy to benefit from a number that tells you when it stops being healthy.

**Do not over-engineer on-call for a low-stakes service.** An internal batch job that runs nightly and can fail silently until morning does not need a 24/7 rotation, a secondary, and a follow-the-sun handoff. It needs a single alert that pages someone during business hours and waits until morning otherwise. Spending the full apparatus of a humane on-call on a service that does not need off-hours coverage is its own kind of waste — and worse, it normalizes off-hours paging for something that never warranted it. The most humane on-call for a service that does not need night coverage is the one that does not page at night at all.

**Do not use on-call to compensate for an unreliable service or noisy alerts.** This is the deepest "when not to" of all. If your answer to thirty pages a week is to grow the rotation so the noise is spread across more people, you have treated a symptom and left the disease. The order of operations is always: fix the noise first, fix the service second, grow the rotation third. A bigger rotation absorbing the same noise is just more people being harmed more gently, and it hides the real problem behind a comfortable-looking schedule. Reducing toil and noise is its own discipline — when [toil, the silent tax on your team](/blog/software-development/site-reliability-engineering/toil-the-silent-tax-on-your-team) ships it covers the systematic side of this; for the architecture-level reliability work that stops the service breaking in the first place, cross-link out to [reliability, SLOs, error budgets, and graceful degradation](/blog/software-development/system-design/reliability-slos-error-budgets-and-graceful-degradation) and [cascading failures, circuit breakers, and bulkheads](/blog/software-development/system-design/cascading-failures-circuit-breakers-and-bulkheads) in the system-design track.

## Key takeaways

- **On-call is a system you design.** A burned-out rotation is a design defect — understaffed, noisy, or uncovered at night — not a willpower failure. You diagnose and fix it like any bug.
- **Rotation frequency is one over team size.** Aim for six to eight people, so each person is primary one week in six to one in eight. Three or fewer burns people out as a matter of arithmetic.
- **A page must never die quietly.** Build an escalation chain — primary, secondary, manager — that auto-escalates on a timer, with a nobody-answered floor and a dead-man's switch. Acknowledgment is the only thing that stops the climb.
- **Kill the 3am page.** Use follow-the-sun if you have the timezones; if you do not, set an explicit nighttime severity threshold so only true emergencies wake a human.
- **Set a page budget — about two incidents per shift.** Breaching it means understaffed, noisy, or unreliable, all fixable. A noisy rotation is a bug to fix, not a tax to pay.
- **Compensate and make it fair.** A flat on-call stipend recognizes the constraint on someone's life; a fair, visible schedule and a real swap mechanism keep the rotation stable.
- **Kill the volunteer hero.** A hero silently absorbing overflow masks the real load and burns out your best person. The pager is a team responsibility, evenly distributed.
- **Make the handoff a ritual.** Open incidents, fragile systems, recent deploys, active silences, and a live pager test — every shift, with a written record.
- **Onboard in stages.** Shadow, runbook tour, reverse-shadow, supported first solo shift. Nobody carries the pager solo on day one.
- **Measure on-call health and act on it.** Pages per shift, off-hours pages, time-to-ack, sleep disruption, and a satisfaction survey you visibly act on. Any red layer is a bug to fix.

## Further reading

- **The Google SRE Book, "Being On-Call"** — the canonical treatment of rotation sizing, the two-incidents-per-shift page budget, and the psychology of sustainable on-call. The source of most of the rules of thumb here.
- **The Google SRE Workbook, "On-Call" and "Implementing SLOs"** — practical detail on operationalizing on-call load and tying paging to SLO burn rates.
- **Alertmanager documentation** — routing, grouping, inhibition, and the receiver model behind the escalation-policy and page-budget artifacts in this post.
- **Prometheus recording and alerting rules documentation** — for the `oncall:pages_per_week` recording rule and the page-budget meta-alert.
- [Reliability is a feature, not an accident: the SRE mindset](/blog/software-development/site-reliability-engineering/reliability-is-a-feature-the-sre-mindset) — the series intro and the define-measure-budget-respond-learn loop this post lives inside.
- [Alerting that doesn't cry wolf](/blog/software-development/site-reliability-engineering/alerting-that-doesnt-cry-wolf) — the sibling on symptom-based, burn-rate alerting that is the upstream fix for a noisy rotation; the page budget is meaningless without it.
- [Reliability, SLOs, error budgets, and graceful degradation](/blog/software-development/system-design/reliability-slos-error-budgets-and-graceful-degradation) — the architecture-time companion: designing the service so it breaks less, which is the deepest fix for an overloaded rotation.
- [Cascading failures, circuit breakers, and bulkheads](/blog/software-development/system-design/cascading-failures-circuit-breakers-and-bulkheads) — how to stop one dependency failure from becoming the multi-page incident that blows your page budget.
