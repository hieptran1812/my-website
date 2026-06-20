---
title: "Case Study: The Bug That Only Happened on Tuesdays"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Follow one intermittent production failure from a confusing 2am page all the way to a verified fix, applying the whole debugging method end to end: observe the pattern, reproduce it deterministically, falsify four wrong theories, bisect code and time, find a three-way conjunction of causes, and prevent it forever."
tags:
  [
    "debugging",
    "software-engineering",
    "case-study",
    "intermittent-bugs",
    "timezone",
    "root-cause-analysis",
    "reproduction",
    "git-bisect",
    "observability",
    "incident-response",
  ]
category: "software-development"
subcategory: "Debugging"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/case-study-the-bug-that-only-happened-on-tuesdays-1.png"
---

The page came in at 02:14 on a Tuesday. Not the first one. The fourth, if you counted the two that auto-resolved before anyone looked. The text was the same every time: `revenue-rollup job failed: exit 1`. By the time the on-call engineer opened her laptop, squinted at the dashboard, and re-ran the job by hand, it succeeded. No error. Clean run. She closed the incident, wrote "transient, re-ran, green," and went back to sleep. That note — "transient, re-ran, green" — is the single most expensive sentence in this entire story, and we are going to spend the next twelve thousand words earning the right to never write it again.

This post is a war story, but it is also a method. It is the whole series compressed into one bug: a real, gnarly, embarrassing, *intermittent* production failure that wore a disguise — it only happened on Tuesdays — and refused to reproduce on demand. We are going to debug it together, from the first useless symptom to a fix so airtight that the failure becomes structurally impossible. Along the way we will read a stack trace that lies to us, falsify four perfectly reasonable hypotheses, run two bisections (one across code, one across the clock), discover that the root cause is not one bug but a *conjunction* of three individually-defensible decisions that only collide for five minutes a week, ship a fix, prove it with a thousand runs, and then do the unglamorous prevention work that turns "transient, re-ran, green" into a failing CI check. The arc is the spine of the entire series, shown in Figure 1: observe, reproduce, hypothesize, bisect, fix, prevent.

![A vertical stack showing the six-step debugging loop from observe at the top through reproduce, hypothesize, bisect, fix, and prevent at the bottom](/imgs/blogs/case-study-the-bug-that-only-happened-on-tuesdays-1.png)

Everything that follows is a composite. The system, the names, the numbers, and the exact log lines are invented to be realistic — this is an illustrative case assembled from the shape of bugs that recur in real production systems, not a documented account of any one company's incident. But every technique is real, every command runs, and the final root cause — a weekly data-partition rollover racing a timezone-naive query across a daylight-relevant clock boundary — is the kind of three-way collision that takes down real revenue pipelines all the time. If you have ever closed an incident with "couldn't reproduce, will monitor," this story is for you. By the end you will be able to take a failure that happens 0.6% of the time and pin it to the exact second, the exact line, and the exact missing assertion that should have caught it in code review.

If you want the philosophical frame before the gore, read the series opener, [stop guessing — the scientific method of debugging](/blog/software-development/debugging/stop-guessing-the-scientific-method-of-debugging). This case is that essay made flesh.

## 1. Observe: turn "transient" into a pattern

The word "transient" is a confession that you stopped looking. Nothing in a deterministic computer is genuinely transient. A bit either flipped or it didn't; a branch was either taken or it wasn't; a row either existed or it didn't. When we say a failure is transient, we mean *we have not yet found the variable that distinguishes the failing runs from the passing ones*. The entire first phase of debugging an intermittent bug is a hunt for that hidden variable, and the only tool that works is **correlation against everything you can measure**: time, host, input, deploy, load, upstream state, the phase of the moon. Most of those will be noise. One will not.

So before touching a debugger, we pull the data. The `revenue-rollup` job is a nightly batch that reads the previous day's transactions out of a partitioned Postgres table, aggregates them per merchant, and writes a summary row that finance reads at 9am. It is scheduled by a cron entry. It emits a metric, `rollup_job_status`, every time it runs: `1` for success, `0` for failure. Six weeks of that metric live in the time-series database. The first real act of debugging is not "read the code." It is this:

```bash
# Pull every failure event for the rollup job over the last 6 weeks,
# print the UTC timestamp and the weekday name for each one.
promtool query range \
  --start="$(date -u -d '6 weeks ago' +%s)" \
  --end="$(date -u +%s)" \
  --step=60 \
  'rollup_job_status == 0' \
| jq -r '.data.result[].values[][1]' \
| while read epoch; do
    date -u -d "@$epoch" '+%Y-%m-%d %H:%M %a'
  done
```

Nine lines came back. Here they are, exactly as they printed:

```console
2026-05-05 02:11 Tue
2026-05-12 02:08 Tue
2026-05-12 02:13 Tue
2026-05-19 02:06 Tue
2026-05-26 02:14 Tue
2026-06-02 02:09 Tue
2026-06-09 02:12 Tue
2026-06-09 02:17 Tue
2026-06-16 02:10 Tue
```

Read those nine lines slowly, because they are the whole case in miniature. Every single failure is a Tuesday. Every single failure is in the 02:06–02:17 UTC band. Two of the dates (May 12 and June 9) fired twice within a few minutes, which tells us the job retries once and the retry also fails. Six weeks, six Tuesdays, nine failures — and *zero* failures on any of the other thirty-six days. This is no longer transient. This is the most non-random thing you will ever see. The work of Figure 2 is to make that transformation explicit: a pile of "random" alerts becomes one precise, testable claim once you bucket the timestamps by hour and by weekday and notice they collapse onto a single point.

![A branching graph showing raw alerts splitting into hour-bucketed and weekday-bucketed views that both converge on a single Tuesday 02:00 UTC signal](/imgs/blogs/case-study-the-bug-that-only-happened-on-tuesdays-2.png)

That two-axis bucketing — by hour *and* by weekday — is the move that matters. A naive engineer buckets by one dimension, sees "all in the early morning," shrugs, and blames "the batch window." A careful engineer cross-tabulates. Here the cross-tab is brutally clear: the failures are not spread across early mornings, they are nailed to *Tuesday* early mornings. The conjunction of two conditions (this weekday AND this hour) is the first hint — though we don't know it yet — that the root cause itself will be a conjunction.

#### Worked example: estimating how long "monitor and hope" would have taken

Before we go further, let's quantify why the "transient, re-ran, green" approach is a trap, because the math is genuinely motivating. The job runs once a day, 7 days a week, so 7 runs a week. It fails on 1 of those 7 days. So the base failure rate per run is roughly $1/7 \approx 14\%$ — but that is *only if you happen to look on a Tuesday*. If you sample the job's health by re-running it at a random time on a random day, your chance of catching the bug on any single manual re-run is far lower, because the failure also requires the narrow 02:06–02:17 window. A manual re-run at 09:00 will always pass. That is *why* the on-call engineer's re-run always succeeded: she was re-running outside the trigger window, so she was sampling from the 99.4% of the schedule where the bug cannot fire.

Put numbers on it. There are $7 \times 24 \times 60 = 10{,}080$ minutes in a week. The failure window is about 12 minutes on one specific weekday — call it 12 minutes out of 10,080, or roughly $0.12\%$ of all wall-clock minutes. If you "monitor and re-run when it breaks" without ever looking at the timestamp pattern, the probability you stumble onto the trigger condition by accident in any given manual investigation is around 1 in 840. You could babysit this job for two years and never see it fail in front of you. The correlation query above found the pattern in under a minute. **The cheapest debugging tool in existence is `GROUP BY`.** This is the core lesson of [reproduce it first, or you're not debugging](/blog/software-development/debugging/reproduce-it-first-or-youre-not-debugging): you cannot fix what you cannot see fail, and the first job is to find the conditions under which it *always* fails.

## 2. Read the stack trace — and notice where it lies

Before we read the error, it's worth understanding *why* a batch job is such fertile ground for a bug like this — the mechanism that lets it hide for weeks. An interactive request handler that fails 0.6% of the time gets noticed fast: a thousand requests a minute means six failures a minute, a user complaint within the hour, an error-rate alert that screams. A nightly batch job that fails 0.6% of the time fails once every 167 days *of runs* — except this one's failures aren't spread uniformly, they're clustered on Tuesdays, so it's actually once a week. Either way, a batch job runs *rarely* (once a day here), so the absolute number of failures is tiny — nine in six weeks — and each one is a single event in the night that auto-resolves before a human engages. There is no flood of errors to force attention; there's a trickle, and a trickle gets rationalized as "transient." The low *frequency* of the job is precisely what lets a real, reproducible, 100%-on-Tuesday bug masquerade as a rare flake. The fix for that blind spot is structural: emit a status metric on every run (which, thankfully, this job did), so even a once-a-day job leaves a trail you can correlate. Without that metric, this investigation would have started from nothing but four terse on-call notes. The lesson, which we'll bank for the prevention phase, is that *rare events need durable evidence*, because you cannot debug what left no trace.

We have a pattern. Now we want the actual error. The on-call notes said "exit 1" with no detail, which means the failure happened somewhere the logging was thin. We go to the job's logs, filter to one of the failing runs, and find the stack trace. Here is the relevant excerpt from the May 19 failure, lightly trimmed:

```log
2026-05-19 02:06:41 UTC INFO  rollup: starting run for date=2026-05-18
2026-05-19 02:06:41 UTC INFO  rollup: querying partition for window
2026-05-19 02:06:41 UTC ERROR rollup: query failed
Traceback (most recent call last):
  File "/app/rollup/job.py", line 142, in run
    rows = self.fetch_transactions(window_start, window_end)
  File "/app/rollup/job.py", line 207, in fetch_transactions
    cur.execute(SQL_FETCH, (window_start, window_end))
psycopg2.errors.UndefinedTable: relation "txn_2026_w20" does not exist
LINE 3:   FROM txn_2026_w20
               ^
```

Now we are getting somewhere, and also somewhere misleading. The proximate error is crisp: `relation "txn_2026_w20" does not exist`. The job tried to read a table named `txn_2026_w20` and Postgres said that table is not there. The `w20` is a week number — week 20 of 2026. So the transactions table is partitioned *by ISO week*, one child table per week, and the job is reaching for the week-20 partition and finding nothing.

Here is where the stack trace lies — not by being false, but by being shallow. It tells you *where* the program died (line 207, the `cur.execute`) and *what* the database said (no such table). It does not tell you *why the table was missing*, which is the actual bug. A junior engineer reads `UndefinedTable`, concludes "someone forgot to create the partition," writes a `CREATE TABLE IF NOT EXISTS`, ships it, and moves on. They will be paged again next Tuesday, because they fixed the symptom's symptom. The stack trace is the *start* of the investigation, not the end of it. The skill is to read it as a witness statement: accurate about what it saw, silent about everything upstream of the crash. (The series post [hypothesize and falsify, not stare and hope](/blog/software-development/debugging/hypothesize-and-falsify-not-stare-and-hope) is the whole discipline of not stopping here.)

What the trace *does* give us, that is genuinely load-bearing, is the table name. `txn_2026_w20`. Cross-reference that with the failure dates. May 19, 2026 is the Monday that *starts* ISO week 21. Wait — week 21? But the job that failed at 02:06 on May 19 was rolling up data for `date=2026-05-18`, which is Sunday, the *last day of ISO week 20*. So at 02:06 UTC on the morning of May 19, the job correctly wants the week-20 partition for Sunday's data, and it isn't there. The partition for the week that *just ended* is missing — but only for a few minutes, only on the day the week ticks over.

That last clause is a hypothesis sneaking in early, and we should be honest that it's a hypothesis, not a fact. The disciplined move is to write down everything the evidence *forces* and separate it from everything we're *guessing*. The evidence forces: (1) the job dies on a missing weekly partition, (2) only on Tuesdays in a 12-minute window. Everything else — *why* the partition is missing in that window — is a theory to be tested, not a conclusion to be assumed.

## 3. Reproduce: control the clock before you touch the code

Here is the iron law of intermittent bugs, and it is non-negotiable: **you do not get to start fixing until you can make the bug happen on demand.** A bug you cannot reproduce is a bug you cannot prove you fixed. You'll change something, the failure won't recur for a week, and you'll declare victory — and then it pages you again, because all you did was perturb the timing. Reproduction is the bright line between debugging and superstition.

The trouble is that this bug's trigger is *time itself*. It fires at 02:06 UTC on a Tuesday. We cannot wait until Tuesday; we cannot ask Postgres to believe it is 02:06 when it is 14:30 on a Thursday. So we do what every serious time-bug investigation does: we **make the clock a controllable input.** In Python, the cleanest way is the `freezegun` library, which monkeypatches `datetime.now`, `time.time`, and friends so the whole process believes it is whatever instant you choose.

The job computes its query window from "now." So if we freeze "now" to a failing instant and re-run, we should see the failure — *if* our understanding of the trigger is right. That last conditional is the point: a reproduction attempt is also a hypothesis test. Here is the first reproducer:

```python
# repro_clock.py — try to make the rollup fail by controlling "now".
import freezegun
from rollup.job import RollupJob

# 2026-05-19 02:06 UTC: a known-failing instant from the metric query.
FAILING_INSTANT = "2026-05-19T02:06:41+00:00"

def attempt(instant: str) -> str:
    with freezegun.freeze_time(instant):
        job = RollupJob(db=connect_test_db())
        try:
            job.run()
            return "PASS"
        except Exception as e:
            return f"FAIL: {type(e).__name__}: {e}"

if __name__ == "__main__":
    print(instant := FAILING_INSTANT, "->", attempt(FAILING_INSTANT))
```

We run it against a copy of the production schema. And it... passes. `PASS`. The job runs clean at the frozen failing instant. Our first reproduction attempt *fails to reproduce the failure*, which is itself a finding: freezing the clock alone is not sufficient. Something about the *database state* differs between our test copy and production at 02:06 on a Tuesday. The clock is necessary but not sufficient. We have controlled one variable and proven it doesn't fully explain the bug. Good — that's progress, even though it feels like a dead end. (This is the central message of the reproduction post: a reproducer that *doesn't* fire is data, not failure.)

What's different about the database? In our test copy, *all* the partitions exist — we built it from a schema dump that already had `txn_2026_w20`. In production at 02:06 Tuesday, the relevant partition is apparently *not yet there*. So the reproducer needs to control a second variable: the **partition state**. We need to reproduce not just the clock but the *moment in the partition lifecycle*. Let's make the test set up the exact adversarial condition — the partition for the just-ended week is missing — and then run under the frozen clock:

```python
# repro_clock_and_data.py — control BOTH the clock and the partition state.
import freezegun
from rollup.job import RollupJob

def attempt(instant: str, drop_partition: str | None) -> str:
    db = connect_test_db()
    seed_full_schema(db)                 # all partitions present
    if drop_partition:
        db.execute(f"DROP TABLE IF EXISTS {drop_partition}")
    with freezegun.freeze_time(instant):
        try:
            RollupJob(db=db).run()
            return "PASS"
        except Exception as e:
            return f"FAIL: {type(e).__name__}: {e}"

if __name__ == "__main__":
    # Tuesday 02:06 UTC, week-20 partition deliberately absent.
    print(attempt("2026-05-19T02:06:41+00:00", drop_partition="txn_2026_w20"))
```

Run it: `FAIL: UndefinedTable: relation "txn_2026_w20" does not exist`. There it is — the exact production error, on demand, in 80 milliseconds, on a Thursday afternoon. We have crossed the bright line. We can now make the bug happen whenever we want. Everything downstream of this point is real engineering instead of fortune-telling.

But notice what we have *assumed* to get here: we assumed the partition is missing at that instant. That's still a hypothesis about production, not a confirmed fact. We've proven that *if* the partition is missing at a frozen Tuesday instant, the job fails — but we haven't proven that's what actually happens in prod. Holding that distinction in your head — "I can reproduce the failure given an assumption I have not yet verified" — is the difference between a real reproduction and a comforting fiction. We'll verify the assumption in section 6.

## 4. Hypothesize and falsify: kill the theories you like

We have a reliable reproducer and a strong suspicion (partition-missing-at-Tuesday-02:00). But a single suspicion that fits the evidence is dangerous, because *several* theories fit this evidence, and confirmation bias will make you marry the first one. The professional move is to enumerate the competing hypotheses, predict what each one implies, and design the single cheapest experiment that would *falsify* each. You are trying to kill your theories, not confirm them. The one that survives every assassination attempt is the one you bisect. Figure 3 is the scorecard.

![A decision tree rooted at the question of why the failure happens only on Tuesday at 02:00 UTC, with four hypotheses branching to falsified results and one surviving branch](/imgs/blogs/case-study-the-bug-that-only-happened-on-tuesdays-3.png)

Here are the five hypotheses we wrote on the whiteboard at 02:40, and the test that decided each.

**H1: It's a load spike.** Tuesday 02:00 might be a heavy-traffic window — a marketing batch, a weekly export — that starves the job of database connections or times it out, and the "missing table" is a misleading downstream symptom of a connection pool exhaustion. *Prediction if true:* CPU, query latency, and connection-pool saturation should be visibly elevated on Tuesdays at 02:00 versus other days. *Falsifying test:* diff the resource metrics. We pulled CPU, QPS, and pool-checkout-wait for Tuesday 02:00 against the six-week average for the same hour on other days. Flat. Dead flat. No spike, no saturation, latency normal. **H1 falsified.** It was the most boring hypothesis and we are glad it died early, because "it's load" sends teams down a multi-week capacity rabbit hole.

**H2: It's a cron collision.** Maybe a second weekly job is scheduled at 02:00 on Tuesdays — a vacuum, a reindex, a partition-maintenance task — and the two collide, one locking out the other. *Prediction if true:* the scheduler log shows two jobs overlapping at 02:00 on Tuesdays specifically. *Falsifying test:* grep the cron and scheduler logs for that window.

```bash
# Look for any job other than the rollup running in the failure window,
# across all six failing Tuesdays.
for d in 05-05 05-12 05-19 05-26 06-02 06-09 06-16; do
  echo "=== 2026-$d ==="
  grep -E "2026-$d 0[12]:" /var/log/scheduler.log \
    | grep -v "rollup" \
    | grep -iE "started|running"
done
```

The output: the rollup job, and nothing else, in that window, on every failing day. There *is* a weekly partition-maintenance task — but the log shows it runs at **02:05**, and crucially it runs *every* day's maintenance pass, not just Tuesday. So a collision can't explain Tuesday-specificity by itself. **H2 falsified as stated** — but file that 02:05 maintenance job away. It is going to matter enormously. A falsified hypothesis often leaves a clue.

**H3: It's a daylight-saving-time shift.** Time bugs love DST. If anything in the pipeline thinks in local time, a spring-forward or fall-back could shift a boundary by an hour and skip or double a window. *Prediction if true:* the failing dates should cluster around a DST transition, or the failure should appear/disappear when one happens. *Falsifying test:* check the timezone database for DST changes near our dates. US DST in 2026 sprang forward on March 8 and falls back on November 1. *None* of our failures — all May and June — are anywhere near a transition. The failures are perfectly regular, every Tuesday, with no discontinuity. A DST event would have made *one* week anomalous; we see uniformity. **H3 falsified** — and again, mostly. DST is not the *trigger*, but the fact that we even asked tells us a timezone is in play somewhere, and the eventual root will indeed be timezone-shaped. The instinct was right; the specific mechanism was wrong.

**H4: It's a data race on the write path.** Maybe two writers (the ingest service and the maintenance job) race on the partition, and under the wrong interleaving the table is briefly dropped-and-recreated, so a read in the gap sees `UndefinedTable`. *Prediction if true:* a concurrency tool should detect conflicting unsynchronized access, and the failure should depend on *interleaving*, not on the calendar. *Falsifying test:* this one is subtle, because Postgres DDL is transactional, but we modeled the suspected interleaving in a Go harness and ran it under the race detector and a stress loop. The detail of *why a race can produce a torn view of state* — no happens-before edge between the dropper and the reader — is exactly the material in [distributed race conditions and ordering](/blog/software-development/debugging/distributed-race-conditions-and-ordering). We ran it:

```bash
# Model the suspected drop/recreate vs read interleaving, 100k iterations,
# under the Go race detector.
go test -race -run TestPartitionReadDuringMaintenance -count=100000 ./rollup/...
```

`ok` — no race reported, and crucially, the read *never* observed a missing table when the maintenance task used a transaction (which the real one does: `DROP` and `CREATE` are inside a `BEGIN/COMMIT`, so a reader sees either the old table or the new one, never neither). The failure is *not* interleaving-dependent; it is *calendar*-dependent, which a race is not. **H4 falsified.** This was the hypothesis the senior engineer on the call liked most, because races are sexy and "it's Tuesday" is not a thing races do. Killing your favorite theory with evidence is the entire job.

**H5: It's a weekly partition rollover, and the read races the create — but not as a thread race, as a *schedule* race.** The transactions table is partitioned by ISO week. Each Monday→Tuesday boundary, a *new* week begins, and some job must create the new week's partition and finalize the old one. If the rollup job reads the just-ended week's partition *before* the maintenance job has finished its weekly work, the read fails. *Prediction if true:* dropping the relevant partition and running the job at the Tuesday instant should reproduce the failure deterministically (it does — section 3), and the failure should vanish if we run the job a few minutes later, after maintenance completes. **H5 survives every test we can throw at it.** It is the only hypothesis consistent with: Tuesday-only (week boundary), 02:06–02:17 (just before/around the 02:05 maintenance pass), no load spike, no cron collision *of equal jobs*, no DST, no thread race.

Notice the structure of what just happened. We did not "find the bug." We *eliminated* four explanations and promoted one survivor, and we did it with five cheap, fast, falsifiable experiments instead of one expensive guess. Figure 3's whole point is that a falsified hypothesis is not wasted work — it is a fence post that bounds where the truth can hide. The matrix in Figure 6, later, records each suspect against its decisive test so that no one can quietly resurrect a dead theory at hour three of the incident.

## 5. Bisect twice: once across code, once across time

We believe the trigger is "rollup reads the new week's partition before maintenance creates it." Two questions remain. First, *when did this start* — was the system always broken, or did a recent change introduce the gap? Second, *what is the exact instant* the failure begins and ends? Both questions are binary searches, and bisection is the right hammer for both. The deep treatment is in [binary-search your bug with bisection](/blog/software-development/debugging/binary-search-your-bug-with-bisection); here we run it twice, on two different axes. Figure 4 shows both passes.

![A left-to-right timeline showing a git bisect across 1300 commits converging in 11 probes, followed by a time bisect across a six-hour window converging on the 02:00 boundary](/imgs/blogs/case-study-the-bug-that-only-happened-on-tuesdays-4.png)

### 5a. Bisecting the code

The metric history shows the *first* Tuesday failure was May 5. The job had run cleanly for months before that. So *something changed* in the weeks before May 5. We have a `good` point (a commit from mid-April that ran many Tuesdays without failing) and a `bad` point (HEAD, which fails every Tuesday). Between them: about 1,300 commits. Linear inspection is hopeless; bisection finds the culprit in $\lceil \log_2 1300 \rceil = 11$ probes.

The beautiful part is that we already have a deterministic reproducer from section 3, so we can *automate* the bisect with `git bisect run`. We write a script that exits `0` if the commit is good and `1` if it's bad, and let git drive:

```bash
#!/usr/bin/env bash
# bisect_rollup.sh — exit 0 if this commit is GOOD, 1 if BAD, 125 to skip.
set -e

# Some old commits won't even build/import; skip them so they don't
# pollute the search (git treats 125 as "untestable").
pip install -e . >/dev/null 2>&1 || exit 125

# Reproduce the Tuesday condition deterministically and check the outcome.
result=$(python repro_clock_and_data.py)   # prints PASS or FAIL: ...
if echo "$result" | grep -q '^PASS'; then
  exit 0    # good: job survived the adversarial Tuesday condition
else
  exit 1    # bad: job died on the missing partition
fi
```

Then:

```bash
git bisect start
git bisect bad HEAD
git bisect good 9f3c1a2     # mid-April, known-clean Tuesdays
git bisect run ./bisect_rollup.sh
```

The exit codes in that script are not decoration; they are the mechanism that makes an *automated* bisect trustworthy, and getting them wrong silently corrupts the search. `git bisect run` interprets the script's exit status: `0` means good, `1` through `124` (except `125`) means bad, and `125` is the special "I cannot test this commit, skip it." That `125` is the unsung hero of real-world bisection. Over 1,300 commits spanning weeks of development, many old commits will not even import — a dependency was added later, a module was renamed, a migration hadn't run. Without the `pip install -e . || exit 125` guard, every un-importable commit would exit non-zero and git would mark it *bad*, poisoning the search and pointing at the wrong culprit. With the guard, git treats those commits as untestable and routes around them, narrowing the range only on commits it can actually evaluate. The other subtlety: because our test is *deterministic* (the reproducer fails 100% of the time on bad commits and passes 100% on good ones), each probe is a single build, and the search is honest. If the bug had been *flaky* — failing only some of the time on bad commits — a single passing probe could wrongly mark a bad commit good, and the bisect would converge on a lie. The counter-move for that case is to run the reproducer `k` times per probe and exit bad on *any* failure, trading more builds for a correct answer. We didn't need it here, precisely because section 3's reproduction work made the bug deterministic first. *That* is why reproduction comes before bisection in the spine: a non-deterministic test makes bisection lie.

Eleven builds later, git printed the verdict:

```console
2c7e9b41 is the first bad commit
Author: dev <dev@example.com>
Date:   Mon Apr 28 16:22:10 2026 +0000

    rollup: read window in report timezone for finance display
```

Read that commit message and feel the floor drop out. *"read window in report timezone."* Someone changed the rollup job to compute its query window in the **report's local timezone** (`America/New_York`) instead of UTC, so that the finance team's dashboard would show days aligned to New York business days. Entirely reasonable. Finance lives in New York; a "day" should be a New York day. The change passed review, passed tests, shipped, and ran fine for a couple of weeks — until the first week-boundary it interacted with. This is the lesson of bisection that no amount of staring at code gives you: it points at the *commit*, and the commit's *intent* is half the root cause. The other half is what that intent collides with.

### 5b. Bisecting the time

We know the culprit commit. We still want the *exact* instant the failure starts and stops, because the width of that window is itself a clue to the mechanism (a wide window means a slow job; a five-minute window means a five-minute maintenance task). The clock is now a controllable input, so we bisect *time* the same way we bisected commits: pick a midpoint instant, run the reproducer, keep the failing half.

```python
# bisect_time.py — find the exact instant the failure begins.
# Precondition: week-20 partition is absent (the adversarial state),
# and the 02:05 maintenance "creates" it (we model that below).
import freezegun
from datetime import datetime, timezone

LO = datetime(2026, 5, 19, 0,  0, tzinfo=timezone.utc)   # known PASS
HI = datetime(2026, 5, 19, 6,  0, tzinfo=timezone.utc)   # known PASS again (after 02:05)

def fails_at(instant) -> bool:
    db = connect_test_db(); seed_full_schema(db)
    db.execute("DROP TABLE IF EXISTS txn_2026_w20")
    # Model maintenance: the partition exists once wall-clock >= 02:05 UTC.
    if instant >= datetime(2026, 5, 19, 2, 5, tzinfo=timezone.utc):
        db.execute("CREATE TABLE txn_2026_w20 PARTITION OF txn ...")
    with freezegun.freeze_time(instant.isoformat()):
        try:
            RollupJob(db=db).run(); return False
        except Exception:
            return True

lo, hi = LO, HI
while (hi - lo).total_seconds() > 1:
    mid = lo + (hi - lo) / 2
    if fails_at(mid):
        # still failing at mid; the failure region includes mid → search left edge
        hi = mid
    else:
        lo = mid
print("failure begins at:", hi.isoformat())
```

The time bisect converged in about 14 probes (a six-hour window down to one second is $\log_2(21600) \approx 14.4$) and printed: failure begins at `2026-05-19T02:00:00+00:00` and ends at `2026-05-19T02:05:00+00:00`. The failure lives in *exactly* the five-minute gap between 02:00 UTC and the 02:05 maintenance pass. Now the 02:06–02:17 spread in the real metrics makes sense: the job is *scheduled* at 02:00 but takes a few seconds to start and connect, and its single retry lands a few minutes later — sometimes still inside the gap (fails), sometimes just past 02:05 (would pass, but the job had already exited on the first attempt). The width of the window — five minutes — is precisely the duration of the maintenance job's head start that the rollup is racing. The clock told us the size of the race.

## 6. Observe in production: confirm the assumption, localize the hop

We have a reproduction, a culprit commit, and a precise window. But everything since section 3 rests on one *assumed* fact: that in production, the week-20 partition is actually missing during that five-minute window. We modeled it; we never *observed* it. Skipping this confirmation is how teams ship a "fix" for a bug they guessed at. So before fixing anything, we instrument production to catch the next Tuesday in the act. This is where observability earns its keep — see [observability for debugging prod](/blog/software-development/debugging/observability-for-debugging-prod) for the design principles; here we apply them surgically.

We add three things, all cheap, all reversible:

1. A **structured log line** in `fetch_transactions` that records the *computed* window boundaries (both in UTC and in the report timezone), the partition name it's about to query, and a correlation ID. Before this, the log told us the table name only *after* it failed. Now it tells us the inputs *before* the query, so we can see the off-by-one boundary in the data, not just the crash.
2. A **trace span** around the database read, tagged with the partition name and a boolean `partition_exists` checked with a fast `to_regclass`. This turns the question "was the partition there?" from a guess into a recorded fact on every run.
3. A **probe metric**, `rollup_target_partition_exists`, emitted as `1`/`0` right before the query, so the next Tuesday's gap shows up as a dip on a dashboard we can put next to `rollup_job_status`.

```python
def fetch_transactions(self, window_start, window_end):
    part = partition_name_for(window_start)        # e.g. "txn_2026_w20"
    exists = self.db.scalar("SELECT to_regclass(%s) IS NOT NULL", (part,))
    log.info(
        "rollup.fetch",
        extra={
            "corr_id": self.corr_id,
            "window_start_utc": window_start.astimezone(timezone.utc).isoformat(),
            "window_start_local": window_start.isoformat(),   # report tz
            "partition": part,
            "partition_exists": exists,
        },
    )
    metrics.gauge("rollup_target_partition_exists", 1 if exists else 0)
    with tracer.start_span("db.fetch_transactions") as span:
        span.set_tag("partition", part)
        span.set_tag("partition_exists", exists)
        return self.db.execute(SQL_FETCH, (window_start, window_end))
```

We ship this *instrumentation only* — no fix yet — on Friday. We wait for Tuesday. At 02:01:12 UTC the following Tuesday, the structured log lights up, and it is everything we predicted and one thing we hadn't fully appreciated:

```log
{"ts":"...T02:01:12Z","corr_id":"r-7f3a","msg":"rollup.fetch",
 "window_start_utc":"2026-06-23T04:00:00Z",
 "window_start_local":"2026-06-23T00:00:00-04:00",
 "partition":"txn_2026_w26","partition_exists":false}
```

Look closely at the two window-start fields. The report-timezone window start is `2026-06-23T00:00:00-04:00` — midnight on a New York Tuesday. But the *same instant in UTC* is `2026-06-23T04:00:00Z` — 4am UTC. Wait, that's not 02:00. Read it again: the *job fires* at 02:00 UTC (its cron), computes "the day to roll up" in New York time, and the *day boundary* it picks lands such that the partition it needs is the one for the week that, in New York's calendar, just ended — and that partition's creation is gated on a maintenance job keyed to UTC week boundaries. The `partition_exists:false` is the smoking gun: at the moment the rollup ran, the target partition genuinely did not exist in production. Our assumption is now an observed fact. The trace confirms the failing hop is the `db.fetch_transactions` span and nothing upstream. We have localized the failure to the exact line, with production evidence, without ever attaching a debugger to a payments-adjacent process at 2am — which you should not do.

#### Worked example: the conjunction that makes it Tuesday-only

Now we can state the root cause precisely, and it is gorgeous in the way only a three-way collision can be. None of these three facts is a bug on its own. Each was a reasonable decision by a competent engineer. The failure requires *all three to be true in the same five-minute window*, which happens once a week. Figure 5 draws the collision.

![A graph showing three reasonable causes — a weekly partition rollover, a report-timezone query, and a timezone offset — converging into a five-minute gap that produces the crash](/imgs/blogs/case-study-the-bug-that-only-happened-on-tuesdays-5.png)

- **Cause 1 (the data lifecycle):** transactions are partitioned by ISO week. A maintenance job creates the new week's partition and finalizes the prior week's at **02:05 UTC**, every day, but the *new-week creation* only does meaningful work on the day a week rolls over. The team chose 02:05 to sit safely after midnight UTC. Reasonable.
- **Cause 2 (the timezone change):** commit `2c7e9b41` made the rollup compute its window in `America/New_York` so finance sees New York business days. Reasonable.
- **Cause 3 (the offset arithmetic):** because New York is UTC−4 in summer, a New York "day" and a New York "week" are *offset* from the UTC week the maintenance job uses. On the specific weekday when the New York week boundary and the rollup's 02:00 UTC firing line up, the rollup asks for a partition keyed to a week that, from the maintenance job's UTC perspective, has only *just* begun — and whose partition isn't created until the 02:05 pass. For the **five minutes** between the 02:00 firing and the 02:05 maintenance, the partition the rollup wants does not exist.

The reason it's *Tuesday* and not Monday is the offset: midnight New York Monday is 04:00 UTC Monday, which is *after* the 02:05 maintenance pass that already ran at 02:05 UTC Monday — so Monday is safe. The collision only lands in the pre-02:05 window on the day the offset pushes the boundary across the 02:00 firing line, which for this configuration is Tuesday. Three reasonable decisions, one weekly five-minute overlap, one outage. **No single person wrote a bug.** The system grew one in the seams between three correct components. This is the most important sentence in root-cause analysis: production failures are usually *interaction* failures, and looking for the one guilty line or the one guilty person is looking for something that isn't there.

## 7. Stress-testing the understanding before we commit to a fix

A root cause is a theory of the bug. Before we spend engineering effort fixing it, a senior engineer stress-tests the theory against the awkward edges of the real world, because a fix built on a slightly-wrong understanding is worse than no fix — it consumes a deploy cycle and a false sense of safety. So we ask the uncomfortable "what if" questions and check that our model survives each one. This is the difference between *a* explanation and *the* explanation.

**What if it only reproduces under load?** Our reproducer runs against an idle test database. Could production load change the timing in a way our test misses — for instance, could a busy database take longer to create the partition, widening the gap? We checked: the maintenance job's create step is a metadata operation (`CREATE TABLE ... PARTITION OF`) that takes milliseconds regardless of table size, because an empty partition has no rows to move. Load does *not* widen the window; the five minutes is a fixed schedule offset (02:00 firing vs 02:05 maintenance), not a load-dependent duration. Our idle reproducer is faithful. If the window *had* been load-dependent, we'd have needed `stress-ng` or a synthetic load generator running alongside the reproducer — a much harder test to make deterministic, and a sign the root cause involved resource contention we hadn't modeled.

**What if it only happens in production, not staging?** It does — staging never failed, which is exactly why nobody caught it pre-prod. The reason is instructive: staging's maintenance job runs on a different cron (it was set to 00:30 by an engineer who wanted faster feedback during a migration), so in staging the partition is *always* created hours before the rollup runs. The gap that exists in prod simply doesn't exist in staging. This is the classic "works on staging" trap: the environments differ in a variable nobody thought was load-bearing — here, a cron offset. The lesson for the prevention phase is that our new test must *engineer the gap explicitly* (drop the partition) rather than relying on environment timing, because environment timing is precisely what lied to us.

**What if you can't attach a debugger in prod?** We couldn't, and we didn't need to. This bug is a perfect example of why the live debugger is the *wrong* first tool for a production intermittent failure. Attaching `gdb` or `pdb` to the rollup process at 02:00 would have meant either (a) being awake and attached at the exact second on the exact Tuesday — absurd — or (b) leaving a debugger attached for a week, which pauses the process on every breakpoint and risks holding a database transaction open across the SLA. Instead we used the post-mortem evidence (the metric history, the stack trace in the logs) plus *passive* instrumentation (the structured log, the trace span, the probe metric) that we could ship safely and read after the fact. The general rule: **for a rare production bug, prefer recording over interrupting.** A log line, a trace span, or an `rr` recording you replay on your laptop beats a live breakpoint you have to be present to hit.

**What if the bug were a flaky race instead of a clean schedule offset?** It's worth dwelling on the *counterfactual*, because it changes the entire toolset. If H4 (the write race) had been the real cause, our reproducer would have been useless — freezing the clock doesn't reproduce a race, because a race depends on *interleaving*, not on wall-clock. We'd have needed a fundamentally different approach: ThreadSanitizer or Go's `-race` to detect the unsynchronized access, `rr` to record a failing run and replay it deterministically (a race that's a 1-in-a-million interleaving is hopeless to reproduce without record-replay), or a deterministic scheduler that forces the bad ordering. The reason we *could* use a clock-and-data reproducer is that this bug, despite looking like a race ("the read raced the create"), is actually a **schedule race**, not a **thread race** — it's deterministic given the schedule, so freezing the schedule's inputs (clock + partition state) reproduces it every time. Diagnosing *which kind of race you have* — deterministic-given-inputs versus genuinely-nondeterministic-interleaving — is the single most important fork in debugging any "sometimes it fails" bug, because it picks your entire toolchain. The deep treatment of the nondeterministic kind is in [distributed race conditions and ordering](/blog/software-development/debugging/distributed-race-conditions-and-ordering); recognizing that *this* one was the deterministic kind is what let us reproduce it in 80 milliseconds instead of recording a million runs.

**What if there were more than one bug?** A genuinely scary possibility on any incident: you fix the bug you found, and the failures continue because there were *two* causes producing the same symptom. We guarded against this by checking that our single root cause explains *all nine* observed failures, not just one — every failure is a Tuesday, every failure is in the pre-02:05 window, and our model predicts exactly that population with no exceptions and no failures it *fails* to explain. If even one of the nine had been a Wednesday, or one had been at 04:00, we'd have known we had a second cause hiding in the noise. The discipline is: *your root cause must account for the full failure population, or you're not done.* Nine for nine is a clean sheet; eight for nine would have meant keep digging.

Every one of those stress tests passed. Now — and only now — do we commit to the fix.

## 8. The decision record: symptom, suspect, confirming test

Before we fix anything, we freeze the investigation into a single artifact so the on-call rotation, the reviewer, and future-you all see the same reasoning. The most useful shape is a matrix: every suspect against the one experiment that confirmed or killed it. Figure 6 is that record, and it is the single most reusable thing produced in this incident, because the *next* weird intermittent failure starts by copying this table and filling in new rows.

![A matrix with rows for load spike, cron collision, DST shift, write race, and partition rollover, each paired with its confirming test and the falsified or confirmed result](/imgs/blogs/case-study-the-bug-that-only-happened-on-tuesdays-6.png)

| Suspect | Confirming / falsifying test | Cost | Result |
|---|---|---|---|
| Load spike (H1) | Diff CPU, QPS, pool-wait: Tuesday 02:00 vs other days | 10 min | **Falsified** — flat, no spike |
| Cron collision (H2) | Grep scheduler log for overlapping jobs in window | 10 min | **Falsified** — only rollup runs; maintenance at 02:05 noted |
| DST shift (H3) | Check tz database for DST transitions near failing dates | 5 min | **Falsified** — no transition in May/June |
| Write race (H4) | Model drop/recreate vs read under `go test -race`, 100k iters | 30 min | **Falsified** — DDL is transactional, no torn read |
| Partition rollover (H5) | Drop target partition, freeze clock to Tuesday 02:00, run | 20 min | **Confirmed** — fails deterministically |

The table earns its place for a reason that goes beyond tidiness. During the incident, at hour three, someone *will* propose resurrecting H1 ("are we *sure* it's not load? Tuesdays do have that export job"). Without this record, you re-litigate it for twenty minutes. With it, you point at the row: *we diffed the metrics, it's flat, here's the query, move on.* A falsified hypothesis stays dead only if you write down how it died. Compare the two debugging styles directly:

| Dimension | Stare-and-guess | Hypothesize-and-falsify |
|---|---|---|
| Where you start | The code, line by line | The pattern, then a falsifiable claim |
| What a wrong guess costs | Hours, and you don't know it was wrong | Minutes, and you *know* it's eliminated |
| When you stop | When the symptom temporarily vanishes | When one hypothesis survives every test |
| What you can prove afterward | "I changed some things and it stopped" | "Only this cause fits; here are the dead ones" |
| Reusability next time | None | The decision matrix is the template |

## 9. The fix: correct the boundary, and make it idempotent

A real fix does two distinct jobs, and weak fixes do only the first. **Job one: correct the specific defect** — the timezone boundary that lets the rollup ask for a not-yet-created partition. **Job two: make the failure mode harmless even if job one is ever wrong again** — so a missing partition becomes a brief, automatic wait-and-retry instead of an outage. Defense in depth: fix the bug, *and* make the bug class survivable. Figure 7 contrasts the before and after.

![A before-and-after figure contrasting the broken local-time query that fails every Tuesday with the fixed UTC-boundary idempotent job that passes a thousand pinned runs](/imgs/blogs/case-study-the-bug-that-only-happened-on-tuesdays-7.png)

### 8a. Correct the boundary

The defect is that the rollup computes *which partition to read* using a timezone-offset window, while the partition lifecycle is keyed to UTC weeks. The two must agree. The cleanest fix is to **decouple the display timezone from the storage timezone**: read partitions by UTC week (matching how they're created and finalized), and apply the New York day grouping *in the aggregation*, not in the partition selection. Here is the diff, trimmed to the load-bearing lines:

```diff
--- a/rollup/job.py
+++ b/rollup/job.py
@@ class RollupJob:
-    def window(self, now):
-        # BUG: compute the partition window in the report timezone,
-        # which is offset from the UTC week the partitions are keyed on.
-        local = now.astimezone(REPORT_TZ)            # America/New_York
-        start = local.replace(hour=0, minute=0, second=0, microsecond=0)
-        return start - timedelta(days=1), start
+    def window(self, now):
+        # FIX: select partitions on the SAME axis they are created on (UTC week).
+        # The report-timezone grouping is applied later, in aggregate(),
+        # so storage and display no longer disagree on what "this week" means.
+        utc = now.astimezone(timezone.utc)
+        day = utc.replace(hour=0, minute=0, second=0, microsecond=0)
+        return day - timedelta(days=1), day
@@ def fetch_transactions(self, start, end):
-        part = partition_name_for(start)             # could be not-yet-created
+        part = partition_name_for(start)
+        self.ensure_partition(part, start)           # create-if-missing, idempotent
         ...
```

The key idea is `partition_name_for(start)` now uses a UTC-derived `start`, so it selects the week-partition that already exists by the time the job runs, on every day including Tuesday. The finance team still sees New York business days, because the *grouping* moved into the aggregation step where it belongs.

### 8b. Make it idempotent and self-healing

Correctness is necessary but not sufficient, because the partition lifecycle could shift again under us — a future change to the maintenance schedule, a new timezone for a new market, a clock skew. So we add `ensure_partition`: if the target partition is missing when the job needs it, create it (it's cheap, it's a `CREATE TABLE IF NOT EXISTS` of an empty partition) instead of crashing. This makes the job **idempotent** — safe to run any number of times with the same effect — and removes the entire *class* of "read raced the create" failures, not just this instance. Idempotency and safe-retry are exactly the properties [building debuggable systems](/blog/software-development/debugging/building-debuggable-systems) argues you should design in from the start.

```python
def ensure_partition(self, part, window_start):
    # Idempotent: create the empty weekly partition if maintenance
    # hasn't yet. Concurrent-safe via IF NOT EXISTS + advisory lock.
    with self.db.advisory_lock(hash_partition(part)):
        self.db.execute(
            f"CREATE TABLE IF NOT EXISTS {part} "
            f"PARTITION OF txn FOR VALUES FROM (%s) TO (%s)",
            (week_bounds(window_start)),
        )
```

A subtle but important point: we wrap the create in a Postgres advisory lock keyed to the partition name, so the rollup and the maintenance job can't *both* try to create it and have one fail with a duplicate error. That's us closing the very race window we ruled out in H4 — not because it caused *this* bug, but because our fix opens a new write path and we refuse to introduce the race we just spent an hour proving didn't exist. Fixing a bug is the wrong time to plant a new one.

## 10. Verify: run it a thousand times against the Tuesday condition

A fix you haven't proven is a hope. The bar for "fixed" on an intermittent bug is not "it didn't fail when I tried it" — it failed only 0.6% of the time *before*, so one passing run proves nothing. The bar is: **run it against the exact failing condition far more times than the original failure rate, and watch it pass every time.** We have a deterministic reproducer of the *worst case* (partition deliberately absent, clock frozen to the Tuesday 02:00 instant), so we hammer it.

```python
# verify_fix.py — 1000 runs against the exact adversarial Tuesday condition.
import freezegun

FAILS = 0
for i in range(1000):
    db = connect_test_db(); seed_full_schema(db)
    db.execute("DROP TABLE IF EXISTS txn_2026_w20")   # the gap, every time
    with freezegun.freeze_time("2026-05-19T02:06:41+00:00"):
        try:
            RollupJob(db=db).run()
        except Exception as e:
            FAILS += 1
            print(f"run {i}: FAIL {type(e).__name__}: {e}")
print(f"failures: {FAILS}/1000")
```

Before the fix, this script prints `failures: 1000/1000` — every single run dies, because we engineer the gap every time. After the fix, it prints `failures: 0/1000`. Zero. The job now *creates the missing partition and proceeds* instead of crashing. We also ran a softer version with a realistic 50/50 partition presence and a randomized firing time across 00:00–03:00 to mimic real schedule jitter, across 5,000 runs: 0 failures. The flake rate went from "once a week, forever" to "zero in ten thousand adversarial runs." That is a before→after you can defend to a skeptical staff engineer, because it's measured against the *worst* case, not a friendly one.

#### Worked example: how many runs is "enough" to trust a fix?

How confident should 1,000 clean runs make us? A little probability keeps us honest. Suppose, pessimistically, the fix were *not* perfect and some residual failure mode fired with true probability $p$ per run. If we observe $n$ clean runs in a row, the probability of that happening *by luck* despite a real defect is $(1-p)^n$. To be 95% confident the residual rate is below some threshold, we use the rule of three: observing $0$ failures in $n$ runs gives an upper 95% bound on the true rate of about $3/n$. So $1{,}000$ clean runs bounds the residual failure probability at roughly $3/1000 = 0.3\%$ — already below the original 0.6%-per-run-on-Tuesday rate, and far below the *0.12%-of-all-minutes* rate. Push to $10{,}000$ runs and the bound drops to $0.03\%$. The point is not the exact number; it's that "I ran it and it worked" is *not* a proof for a rare bug, and the rule of three tells you exactly how many runs you need to make a defensible claim. For a bug that fired ~1 time in 167 Tuesday-runs, a thousand adversarial passes is a real verification; a single re-run — the original sin of "transient, re-ran, green" — is not.

## 11. Prevent: make it impossible to ship again

Fixing the bug is half the job. The other half is making sure *this class of bug* cannot reach production unnoticed again, and that the system is more debuggable next time. Prevention has three layers: a test that would have failed in CI, an assertion that fails loudly in prod, and a five-whys that fixes the *process* gap that let three time-dependent assumptions ship without a single clock-pinned test. Figure 8 is the five-whys chain.

![A five-whys tree descending from the missing-partition crash through the local-timezone gap to the systemic root that no clock-pinned test existed in CI, with two prevention fixes](/imgs/blogs/case-study-the-bug-that-only-happened-on-tuesdays-8.png)

### 10a. The test that would have caught it

The whole bug existed because *no test ever ran the job at a time boundary*. The unit tests ran at whatever wall-clock the CI runner happened to have, which was never a Tuesday week-rollover with a missing partition. We add a parametrized, clock-frozen boundary test that exercises exactly the conditions we now understand:

```python
import pytest, freezegun

@pytest.mark.parametrize("instant,partition_missing,expect", [
    # The exact bug: Tuesday 02:00 UTC, week partition not yet created.
    ("2026-05-19T02:00:00+00:00", True,  "ok"),
    # Mid-week, partition present: the easy case must still work.
    ("2026-05-21T02:00:00+00:00", False, "ok"),
    # Year boundary, ISO week 1 vs week 53: a classic off-by-one trap.
    ("2026-12-29T02:00:00+00:00", True,  "ok"),
    # DST spring-forward instant: prove the boundary logic survives it.
    ("2026-03-08T07:00:00+00:00", True,  "ok"),
])
def test_rollup_at_time_boundaries(instant, partition_missing, expect):
    db = fresh_db_with_full_schema()
    if partition_missing:
        db.execute(f"DROP TABLE IF EXISTS {target_partition(instant)}")
    with freezegun.freeze_time(instant):
        RollupJob(db=db).run()          # must NOT raise
    assert summary_row_exists(db), expect
```

This test *fails on the old code* — we ran it against the pre-fix commit to confirm it's a real regression test and not a tautology, and it raised `UndefinedTable` on the first parameter, exactly as production did. A test that doesn't fail on the buggy code proves nothing. Now it's a tripwire: any future change that reintroduces a timezone/partition boundary mismatch turns CI red in seconds, on a developer's laptop, weeks before any Tuesday.

### 10b. The assertion that fails loudly in prod

Tests catch what you thought to test. Assertions catch what you didn't. We add a runtime invariant at the top of `fetch_transactions`: the partition the job is about to query *must* exist (after `ensure_partition`), and if it somehow doesn't, fail with a message that contains everything a debugger needs — the partition name, the computed UTC and local windows, and the firing time — instead of a bare `UndefinedTable` ten frames deep:

```python
part = partition_name_for(start)
self.ensure_partition(part, start)
assert self.db.scalar("SELECT to_regclass(%s) IS NOT NULL", (part,)), (
    f"partition {part} missing after ensure_partition; "
    f"window_utc={start.astimezone(timezone.utc).isoformat()} "
    f"window_local={start.isoformat()} fired_at={now_utc().isoformat()}"
)
```

The difference between the old failure and this one is the difference between a five-Tuesday investigation and a five-minute one. The old crash said "table doesn't exist" and made you reverse-engineer *why*. This assertion hands the next engineer the timezone math on a plate. **Good error messages are pre-paid debugging.** This is the heart of building debuggable systems: you cannot prevent every bug, but you can guarantee that the next one announces its own root cause.

### 10c. The five-whys to the systemic root

Finally, the unglamorous part that prevents the *next* class of bug, not just this one. We run a five-whys, and the discipline is to keep asking "why" past the satisfying technical answer until you hit a *process* answer, because process gaps are what let three reasonable engineers ship three reasonable changes that collided. The full method is in [root-cause analysis and the five whys](/blog/software-development/debugging/root-cause-analysis-and-the-five-whys); here is the chain for this incident:

- **Why did the job crash?** The week partition it queried didn't exist.
- **Why didn't it exist?** The job ran (02:00 UTC) before the maintenance pass created it (02:05 UTC), in the five-minute gap.
- **Why did the job run in that gap?** Commit `2c7e9b41` made it compute its window in `America/New_York`, whose offset pushes the week boundary across the 02:00 firing line on Tuesdays.
- **Why did that offset bug ship?** No test exercised the job at a time boundary with a missing partition; the timezone change looked correct in isolation and was reviewed in isolation.
- **Why was there no boundary test?** *(the systemic root)* The team had no convention that any code computing windows, partitions, or schedules must ship with clock-pinned tests at the day/week/year/DST boundaries. Time logic was tested at "now," which is never a boundary.

The fix for the *systemic* root is not a code change at all — it's a checklist item and a CI guard: **any PR that touches time, timezone, partition, or schedule logic must include a frozen-clock boundary test, enforced by a CODEOWNERS rule plus a lint check that flags `datetime.now()`/`astimezone` changes lacking an accompanying `freeze_time` test.** That is what makes the bug *impossible to ship again*, as opposed to merely fixed this once. We also added the `rollup_target_partition_exists` probe metric (from section 6) permanently, with an alert that fires *before* the job crashes — turning a future recurrence from an outage into a warning.

## 12. War story: this exact shape, in the wild

The Tuesday bug is a composite, but its *shape* — a time/calendar boundary interacting with a data lifecycle — is one of the most reliable producers of real outages in the industry. A few real, documented examples of the genus, so you trust that this isn't a contrived puzzle:

**The leap-second cascade (2012, and again 2015, 2016).** When a leap second was inserted (the clock hit `23:59:60` UTC), the Linux kernel's high-resolution timer subsystem hit a code path that left some `hrtimer` state inconsistent, and processes that called `futex`-based waits spun the CPU at 100%. Sites running Java and certain databases — including, famously, a number of large web platforms — saw mass CPU saturation at the exact instant of the leap second. The bug was real, latent for years, and only fired on a calendar event that happens irregularly every few years. The fix many operators used in a panic was a one-liner: `date -s "$(date)"` to re-seat the clock and clear the bad timer state. The shape is identical to our case: a rare calendar boundary meets a state assumption that's true 99.9999% of the time.

**Zune leap-year freeze (December 31, 2008).** Microsoft's Zune media players froze worldwide on the last day of 2008. The cause was a date-conversion loop in the real-time clock driver that, on the 366th day of a leap year, never terminated — an off-by-one in handling the extra day. Every Zune on Earth hit it simultaneously because they all shared one wall clock. It "fixed itself" the next day (January 1, when the day count rolled into a non-leap-affected range), which is the cruelest property of time bugs: they self-heal, so "transient, re-ran, green" feels true. The Zune bug, like ours, only fired at a specific calendar boundary and vanished afterward.

**Knight Capital (2012).** Different mechanism — a deploy that left an old, dormant code path enabled on one of eight servers — but the *debugging* lesson is the same one our case teaches: the failure lived in the *seam* between components that were each individually fine (a feature flag reused for a new purpose, an incomplete deploy, a dormant routine). No single line was "the bug." Knight lost about \$440 million in 45 minutes. The cost of a missing assertion and an incomplete rollout is not abstract.

The throughline across all three: **production failures concentrate at boundaries and seams** — calendar boundaries, week/year rollovers, partial deploys, timezone offsets, the interface between two correct subsystems — and they love to self-heal, which is exactly what makes them feel "transient" and exactly why the correlation-first method matters. If you take one habit from this post, make it this: *when something fails intermittently, bucket the failures by every clock you have before you read a single line of code.*

## 13. How to reach for this (and when not to)

This case used a lot of tools. Most bugs need a small subset, and reaching for the heavy machinery when a `GROUP BY` would do is its own failure mode. A decisive guide:

- **Always start with correlation, never with the code.** Bucket failures by time, host, input, deploy, and version *first*. For an intermittent bug this is non-optional and it is the highest-leverage minute you will spend. If the pattern is obvious (ours was), you've saved days.
- **Make the hidden variable controllable before you fix anything.** For a time bug that's `freezegun`/`libfaketime`; for a data bug it's a seeded fixture; for a concurrency bug it's a deterministic scheduler or `rr` record-replay. If you can't reproduce on demand, you are not debugging, you are perturbing. Don't proceed.
- **Bisect when the question is "what changed?" — and only then.** Bisection needs a clean good→bad transition and a deterministic test. It's overkill for a bug you can read off the stack trace, and it *misleads* on a flaky bug (run each probe k times, exit on any failure). For a regression in 1,300 commits, nothing beats it.
- **Attach a debugger as a last resort in prod, not a first one.** Do *not* attach `gdb` to a payments process under load — you'll pause it and breach an SLA. Prefer a structured log line, a trace span, or a probe metric you can ship safely and read after the fact, as we did in section 6. Reach for the live debugger only when a post-mortem core dump or `rr` replay genuinely isn't enough.
- **Don't chase a heisenbug at the wrong optimization level.** If a bug appears at `-O2` and vanishes at `-O0`, that's a clue about compiler reordering or undefined behavior, not a reason to give up — but reproduce it at the level where your tools work before you theorize.
- **Stop when one hypothesis survives every cheap falsifying test, not when the symptom temporarily disappears.** The disappearing symptom is the trap that wrote "transient, re-ran, green." A bug is fixed when you've run it against its worst case more times than its original failure rate and watched it pass, *and* you have a test that fails on the old code.

For quick reference, here is the tool we reached for at each phase of this incident, what it found, what it cost, and the symptom that should make you reach for it next time:

| Phase | Tool we used | What it found | Overhead | Reach for it when |
|---|---|---|---|---|
| Observe | `promtool` + `GROUP BY` weekday/hour | The Tuesday-02:00 pattern | Seconds | A failure is intermittent and you have *any* metric history |
| Read | Stack trace + log grep | The missing-partition proximate error | Seconds | Any crash with a logged traceback |
| Reproduce | `freezegun` clock + seeded data | Deterministic on-demand failure | Minutes | The trigger is time, date, or a specific data state |
| Hypothesize | Metric diffs, log greps, `go test -race` | Four falsified theories, one survivor | 5–30 min each | More than one cause fits the evidence |
| Bisect | `git bisect run` + the reproducer | Culprit commit in 11 probes | One build per probe | The question is "what change introduced this?" |
| Bisect (time) | Hand-written binary search over instants | The exact five-minute window | 14 probes | You want the precise boundary, not just the day |
| Localize | Structured log + trace span + probe metric | Confirmed the partition gap in prod | Negligible, safe in prod | You must observe prod without interrupting it |
| Verify | 1,000-run adversarial loop | 0 failures, rule-of-three bound | Minutes | Proving a rare bug is actually fixed |

The pattern across the table: the cheap, safe, passive tools (correlation, log grep, structured logging) do most of the work, and the expensive, intrusive tools (live debuggers, race detectors, record-replay) are reserved for the specific symptom that demands them. A junior engineer reaches for the debugger first; a senior engineer reaches for `GROUP BY` first and earns the right to the debugger only when the cheap tools run out. The whole series, gathered into a single decision flow, lives in [the debugging playbook capstone](/blog/software-development/debugging/capstone-the-debugging-playbook); this case is one worked path through it.

When is the full method *not* worth it? When the fix is genuinely trivial and the blast radius is tiny — a typo in a log string, a missing null check the linter already flagged. Don't run a five-whys on a typo. But the moment a failure is *intermittent*, *production*, or *expensive*, every step here pays for itself many times over. The Tuesday bug cost four 2am pages and a finance team that didn't trust the 9am report for six weeks; the full investigation took one focused day. That trade is not close.

## Key takeaways

- **"Transient" means you stopped looking.** Nothing in a deterministic system is random; there is always a hidden variable distinguishing failing runs from passing ones. Find it by correlating failures against every clock and dimension you can measure, *before* reading code.
- **The cheapest debugging tool is `GROUP BY`.** Six weeks of a status metric, bucketed by hour and weekday, turned "random flake" into "Tuesday 02:00 UTC" in under a minute and saved a potential two-year babysitting effort.
- **Reproduction is the bright line.** Make the hidden variable controllable — the clock, the data, the interleaving — until the bug fails *on demand, every time*. A bug you can't reproduce is a bug you can't prove you fixed.
- **Falsify the hypotheses you like.** We killed load, cron collision, DST, and a write race — each with one cheap experiment — and the survivor was the one nobody found exciting. Write down how each theory died so it stays dead.
- **Bisect on more than one axis.** Git bisect found the culprit commit in 11 probes; a time bisect found the exact five-minute failure window in 14. The same logarithmic search works on code, data, config, and the clock.
- **Production failures are interaction failures.** The root was a conjunction of three individually-reasonable decisions — a UTC partition lifecycle, a report-timezone query, and a summer offset — that collide for five minutes a week. Stop hunting for the one guilty line; look at the seams.
- **Verify against the worst case, not a friendly one.** Zero failures in a thousand runs *with the partition deliberately missing every time* is a defensible fix; one passing re-run is the original sin. The rule of three tells you how many runs you need.
- **Prevent at three layers.** A clock-pinned boundary test that fails on the old code, a loud assertion that hands the next engineer the root cause, and a five-whys that fixes the *process* gap (no time-boundary test convention) so the whole class can't ship again.

## Further reading

- The series opener that frames every step here as one method: [stop guessing — the scientific method of debugging](/blog/software-development/debugging/stop-guessing-the-scientific-method-of-debugging).
- Why you must reproduce before you fix, and how to make intermittent bugs deterministic: [reproduce it first, or you're not debugging](/blog/software-development/debugging/reproduce-it-first-or-youre-not-debugging).
- Turning suspicions into cheap falsifiable experiments: [hypothesize and falsify, not stare and hope](/blog/software-development/debugging/hypothesize-and-falsify-not-stare-and-hope).
- The logarithmic search we ran twice, on code and on time: [binary-search your bug with bisection](/blog/software-development/debugging/binary-search-your-bug-with-bisection).
- Shipping logs, traces, and probe metrics that catch the next Tuesday in the act: [observability for debugging prod](/blog/software-development/debugging/observability-for-debugging-prod).
- Why the write-race hypothesis was even plausible, and how DDL transactionality killed it: [distributed race conditions and ordering](/blog/software-development/debugging/distributed-race-conditions-and-ordering).
- Asking "why" past the technical answer to the process root: [root-cause analysis and the five whys](/blog/software-development/debugging/root-cause-analysis-and-the-five-whys).
- Idempotency, loud assertions, and the design habits that make the next bug announce itself: [building debuggable systems](/blog/software-development/debugging/building-debuggable-systems).
- The whole method on one page, with this case as a worked entry: [the debugging playbook (capstone)](/blog/software-development/debugging/capstone-the-debugging-playbook).
- David Agans, *Debugging: The 9 Indispensable Rules* — "Quit Thinking and Look" and "Make It Fail" are this post in two sentences.
- Andreas Zeller, *Why Programs Fail* — the formal treatment of delta debugging, the bisection-of-everything idea generalized.
