---
title: "Binary-Search Your Bug With Bisection: The Most Underused Technique in Debugging"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Learn to halve your search space every step so a regression hiding in four thousand commits falls in twelve builds, and discover that the same logarithmic search localizes faults in data, config, dependency versions, time, feature flags, and a chain of services."
tags:
  [
    "debugging",
    "software-engineering",
    "git-bisect",
    "binary-search",
    "regression",
    "bisection",
    "root-cause-analysis",
    "troubleshooting",
    "automation",
  ]
category: "software-development"
subcategory: "Debugging"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/binary-search-your-bug-with-bisection-1.png"
---

A regression shipped last Tuesday. Nobody noticed until Friday, when a customer wrote in to say the export that used to finish in four seconds now takes ninety. You pull up the dashboard and there it is, ugly and undeniable: p95 export latency was flat at four seconds for months, and then sometime in the last week it stepped up to a wall. No error. No crash. No log line that says "this got slow." Just a number that used to be small and is now large. And between "it was fine" and "it is broken" sit four thousand and ninety-six commits, because your team is busy and merges all day, and any one of them could be the one that did it.

This is the moment where most engineers reach for their intuition and lose the afternoon. They open the export code. They read it. They guess that it must be the new caching layer, or the database driver upgrade, or that refactor someone did to the serializer. They `git log` and squint at commit messages looking for a smoking gun, as if the author of the bug labeled their commit "make export slow." They check out a commit that "feels recent enough," build it, run it, and it is slow, so they check out an older one, and it is slow too, so they go older still — a linear march backward through history, one build at a time, burning ten minutes per build, getting more frustrated with each one. By the time they have tested twenty commits they have spent three hours and ruled out half a percent of the candidates.

There is a better way, and it is not cleverness, it is arithmetic. The number 4,096 is two to the twelfth power. If, instead of walking commits one at a time, you could ask "is the bug present at *this* commit, yes or no?" and use the answer to throw away *half* the remaining commits every time, you would find the exact commit that introduced the regression in twelve tests. Not twelve hundred. Twelve. The figure below is the entire idea: every probe you make cuts the suspect pool in half, and halving four thousand things a dozen times leaves you with one.

![A vertical stack showing four thousand suspects narrowing through repeated midpoint probes down to a single culprit commit in twelve halving steps](/imgs/blogs/binary-search-your-bug-with-bisection-1.png)

This is **bisection**, and it is the most powerful debugging technique that most working engineers underuse. The word comes from "bisect," to cut in two; the idea is binary search, the same algorithm you learned for finding a number in a sorted array, pointed at the problem of finding a bug. By the end of this post you will be able to do several concrete things. You will run `git bisect` by hand to localize a regression to a single commit across a huge range. You will write a `git bisect run` script that does the entire search unattended while you get coffee, finding the first bad commit over a weekend's worth of work in minutes. You will recognize that *anything* you can split in half and test — a dataset, a config file, a list of dependency versions, a window of time, a chain of services — is binary-searchable, and you will bisect all of them. And you will handle the cases that break naive bisection: the flaky bug that comes and goes, the commit in the middle of the range that does not even build, the baseline you thought was good but was not. This is the **bisect** stage of the series' master loop — observe, reproduce, hypothesize, bisect, fix, prevent — and it is the stage where a search that could take days takes minutes instead. If you have not read the series intro, [the scientific method of debugging](/blog/software-development/debugging/stop-guessing-the-scientific-method-of-debugging) frames the whole loop, and [hypothesize and falsify, not stare and hope](/blog/software-development/debugging/hypothesize-and-falsify-not-stare-and-hope) is the engine that makes each probe a real experiment. Bisection is that engine running on autopilot.

## 1. The mechanism: why halving works and why it is so fast

Let me make the speed concrete, because the gap between linear and logarithmic search is the entire reason bisection feels like magic the first time you use it. Suppose you have N suspects and a test that tells you, for any point in the range, whether the bug is present at or before that point. A linear search checks them one at a time: in the worst case you do N tests, and on average N over two. With N equal to 4,096, that is up to four thousand builds. A binary search does something fundamentally different. It tests the *middle*. The answer to that single test — bug present or not — eliminates not one suspect but *half* of them. Then it tests the middle of the surviving half, eliminating half of *that*, and so on. The number of suspects after k tests is N divided by two to the k, and you are done when that hits one. Solving for k gives the punchline: k equals the base-two logarithm of N. For 4,096, that is 12. For a million, it is 20. For a billion, it is 30.

That is not a small constant-factor win, it is a change in growth class, and it is why the comparison below is not a fair fight. Linear search grows in proportion to the size of the problem; bisection grows in proportion to the *logarithm* of the size of the problem. Doubling the number of commits adds *one* step to a bisection. You could have ten times as many commits and bisection would only need three or four more probes. This is the same reason a phone book lookup is fast even though the book has a million names: you open to the middle, decide left or right, and repeat.

![A two column before and after contrast of testing commits one at a time against bisecting the same range, showing four thousand builds collapsing to twelve probes](/imgs/blogs/binary-search-your-bug-with-bisection-2.png)

The before-and-after above is the whole economic argument. On the left, the linear scan: test commit by commit, worst case 4,096 builds, hours to days, and in practice people give up and ship a guess. On the right, bisection: probe the midpoint, twelve builds total because twelve is the base-two log of 4,096, and the first bad commit found in minutes. If each build-and-test cycle takes five minutes, the linear scan's worst case is over three hundred hours and bisection is one hour. That is the difference between a quarter of a year of compute and a lunch break.

But — and this is the load-bearing assumption that the rest of this post keeps coming back to — bisection only works if the property you are testing is **monotonic** along the axis you are searching. Monotonic means: once the bug appears, it stays. If you order your commits from oldest to newest, there must be a single point where "good" flips to "bad" and never flips back. Every commit before that point is good; every commit at or after it is bad. If that is true, one test in the middle tells you which half the transition lives in, and you can recurse. If it is *not* true — if the bug appears, disappears, reappears — then the midpoint test lies to you, because "bad here" no longer means "bad in this whole half." Almost everything that goes wrong with bisection in practice is a violation of monotonicity, and almost every fix is a way to restore it. Hold that thought; section 9 is entirely about it.

Here is why monotonicity is so often *true* for regressions, which is what makes bisection so practical. A regression is, by definition, a change that turned working code into broken code. There is a commit that introduced it. Before that commit the behavior was correct; the commit changed something — a default, a line of logic, an upgraded dependency — and from then on the behavior is wrong. That is the textbook monotonic transition. The export was fast for every commit up to the one that changed the default batch size from 1,000 to 1, and slow for every commit after. Bisection was practically designed for this shape of problem, which is why `git bisect` exists as a first-class command.

## 2. `git bisect` by hand, step by step

Let me walk through an actual `git bisect` session, because the commands are simple and worth building fluency with before we automate them. The setup: you know the current `HEAD` is bad (the export is slow), and you know that the release tag `v2.3` from three weeks ago was good (the export was fast). Everything between them is suspect.

You start the bisect, tell git which commit is bad and which is good, and git takes over:

```bash
$ git bisect start
$ git bisect bad                 # current HEAD is bad; no arg means HEAD
$ git bisect good v2.3           # this tag was known-good
Bisecting: 2047 revisions left to test after this (roughly 11 steps)
[a1b2c3d] Refactor the export serializer
```

Read that last line carefully, because it is doing a lot. Git has computed the set of commits reachable from `HEAD` but not from `v2.3` — that is the search range — counted them (4,095 commits, so 2,047 on each side of the midpoint), picked the commit that most evenly splits that set, and **checked it out for you**. Your working tree is now sitting at commit `a1b2c3d`, the midpoint. Git even tells you it expects about 11 more steps, which is the base-two log of the range. Your job now is to answer one question: is the bug present *here*?

So you build and test at this commit. You run the export, you time it. Say it is slow — ninety seconds, the bug is present:

```bash
$ make build && ./run-export --time
export completed in 90.2s
$ git bisect bad
Bisecting: 1023 revisions left to test after this (roughly 10 steps)
[d4e5f6a] Bump the database driver to 5.2
```

You marked it bad. Git threw away the entire *newer* half — every commit after `a1b2c3d` is now irrelevant, because the bug was already present at `a1b2c3d`, so whatever introduced it is at `a1b2c3d` or earlier. Git checked out the midpoint of the remaining older half and is now down to 1,023 revisions. You test again. Say this one is fast — four seconds, the bug is *not* present:

```bash
$ make build && ./run-export --time
export completed in 3.9s
$ git bisect good
Bisecting: 511 revisions left to test after this (roughly 9 steps)
[7a8b9c0] Add retry wrapper around fetch
```

This time you marked it good, so git threw away the entire *older* half up to and including `d4e5f6a` — the bug was not present there, so it must have been introduced *after* it. Notice the rhythm: every answer, good or bad, halves the range. You keep going — build, time, mark — and the "revisions left" number keeps dropping: 511, 255, 127, 63, 31, 15, 7, 3, 1. When git has narrowed it to a single commit, it announces the culprit:

```bash
$ git bisect bad
9f8e7d6 is the first bad commit
commit 9f8e7d6
Author: Some Engineer <eng@example.com>
Date:   Tue 14:22

    Tune default export batch size for memory

 src/export/config.py | 2 +-
 1 file changed, 1 insertion(+), 1 deletion(-)
```

There it is: a two-line change to a config default. The author was trying to reduce memory pressure and dropped the batch size, which exploded the number of round trips and tanked latency. You would never have found this by reading the export code — the export code did not change, the *default it reads* changed. Bisection found it without you understanding anything about the bug; it just kept asking "bad or good?" until one commit was left.

When you are done, you clean up:

```bash
$ git bisect reset
Previous HEAD position was 9f8e7d6 ...
Switched to branch 'main'
```

`git bisect reset` returns your working tree to where you were before you started — back on your branch, at the commit you were on. This matters because during a bisect, git has been checking out arbitrary historical commits, and `reset` is how you get back to the present. Forgetting it is the single most common "wait, why is my repo in a weird state?" after a bisect.

## 3. The mechanism underneath: it is binary search over a DAG, not a list

It is tempting to picture commit history as a straight line, but real history is a **directed acyclic graph** — a DAG. Commits have parents; a merge commit has two (or more) parents; branches diverge and rejoin. So when I said git "picks the midpoint," I glossed over something genuinely clever, and it is worth understanding because it explains why `git bisect` handles merges correctly when a naive "test the middle of the list" would not.

![A branching graph in which a known good commit fans out into a feature branch and a main line that join at a merge commit before reaching the bad current head, with git picking a midpoint that splits the reachable set](/imgs/blogs/binary-search-your-bug-with-bisection-3.png)

The graph above shows the situation: a known-good commit, a feature branch of 200 commits and a main line of 300 commits that both descend from it, a merge commit where they rejoin, and the bad `HEAD` beyond. There is no single "middle commit" of a graph the way there is a middle element of an array. So git does something smarter. For each candidate commit in the suspect range, it counts how many commits are *reachable* from it within the range — that is, how many ancestors of it are still suspects. A commit that splits the range into two equal halves by reachable-commit count is the ideal probe, because testing it eliminates as close to half the remaining commits as possible regardless of the branching structure. Git finds the commit whose reachable set is closest to half the total and checks *that* out. The formal name for what it is computing is the commit that best bisects the DAG by the number of ancestors.

Why does this matter to you? Two reasons. First, it means you can trust `git bisect` on a messy real-world history full of merges — it is not fooled by the branching, because it counts reachable commits, not positions in a list. Second, it explains a subtlety: when git marks a commit good, it eliminates that commit *and all of its ancestors* (everything reachable from it), because "good" means the bug is not present in any of the history leading up to that commit. When git marks a commit bad, it eliminates that commit and all of its *descendants* within the range. On a linear history those are just "older half" and "newer half." On a DAG with merges, "ancestors" and "descendants" can be large, overlapping sets, and the reachable-count math is what makes the elimination correct. The number "revisions left" that git prints after each step is the size of the surviving suspect set, and it is genuinely the count of commits that could still be the culprit, not an approximation.

There is one more graph-related detail that surprises people: the first bad commit `git bisect` reports is the first commit, in topological order, at which the bug is present and whose parents are all good. On a branchy history there can be more than one path to the bug, but bisection finds the boundary correctly because it reasons over reachability, not over a flattened log. If a regression was introduced *on a branch* and only became visible after the merge, bisection can land on either the branch commit that introduced it or the merge commit that exposed it, depending on where the "good or bad" answers actually flip — which is exactly correct, and a hint that you should test the reported commit's parents to understand which.

## 4. The killer feature: `git bisect run` for full automation

Everything so far has been you, sitting at the keyboard, building and testing and typing `git bisect good` or `bad` a dozen times. That is fine for twelve steps. But the real power of bisection is that the "is the bug present here?" question is usually something a *script* can answer, and if you can write that script, `git bisect run` will do the entire search unattended. You hand git a command; git checks out the midpoint, runs your command, looks at its **exit code**, decides good or bad from that, and recurses — all the way down to the first bad commit — without you touching the keyboard.

The protocol is the contract you must honor in your script's exit code:

- **Exit 0** means *good* (the bug is not present at this commit).
- **Exit 1 through 124, plus 126 and 127** mean *bad* (the bug is present).
- **Exit 125** means *skip* (this commit cannot be tested — it does not build, the test harness is broken here — so do not count it for or against, just move on).
- **Exit 128 and above** abort the bisect (a special signal that something is catastrophically wrong).

So your whole job is to write a script that builds the code, reproduces the bug, and exits 0 if the bug is absent and non-zero if it is present. Here is a complete, real `bisect run` script for the slow-export regression. It builds, runs the export, captures the elapsed time, and decides based on a threshold:

```bash
#!/usr/bin/env bash
# bisect-export.sh — exit 0 if export is fast (good), 1 if slow (bad), 125 if unbuildable (skip).
set -u

# Build at the current commit. If the build fails, this commit is untestable: skip it.
if ! make build > /tmp/build.log 2>&1; then
    echo "build failed at $(git rev-parse --short HEAD), skipping"
    exit 125
fi

# Run the reproducer and capture wall-clock seconds.
elapsed=$(./run-export --time --quiet 2>/dev/null | awk '/completed in/ {print $3}' | tr -d 's')

# Defensive: if we could not even get a number, the repro is broken here — skip.
if [ -z "$elapsed" ]; then
    echo "could not measure export at $(git rev-parse --short HEAD), skipping"
    exit 125
fi

# The decision boundary. Good was ~4s; bad was ~90s. Anything over 30s is the regression.
if (( $(echo "$elapsed > 30.0" | bc -l) )); then
    echo "SLOW: ${elapsed}s — bad"
    exit 1
else
    echo "fast: ${elapsed}s — good"
    exit 0
fi
```

You make it executable, then launch the whole search with three commands:

```bash
$ chmod +x bisect-export.sh
$ git bisect start
$ git bisect bad
$ git bisect good v2.3
$ git bisect run ./bisect-export.sh
```

Now you walk away. Git checks out the midpoint, runs `bisect-export.sh`, reads the exit code, marks the commit good or bad or skips it, checks out the next midpoint, and repeats — twelve times — until it prints the first bad commit and the diff that introduced it. Twelve builds, each maybe five minutes, is an hour of unattended compute that finds a needle in four thousand commits while you do something else. The first time you watch `git bisect run` march through a couple thousand commits and land on a one-line change, it changes how you think about regressions forever.

![A left to right timeline of twelve automated probes each halving the remaining commit range from four thousand down to a single first bad commit](/imgs/blogs/binary-search-your-bug-with-bisection-4.png)

The timeline above is what `git bisect run` does while you are away: probe 1 cuts 4,096 to 2,048, probe 3 is down around 1,024 to 512, probe 6 around 128 to 64, probe 9 around 16 to 8, probe 11 around 4 to 2, and probe 12 lands on the single first bad commit. Each tick is one automated build-and-test. Nothing about this requires you to understand the bug; it requires you to be able to *detect* the bug from a script's exit code, which is a much lower bar.

#### Worked example: a performance regression bisected over 4,096 commits in exactly 12 builds

Let me put real numbers on the export case, because the arithmetic is the entire point. The known-good tag `v2.3` and the current bad `HEAD` had exactly 4,096 commits between them — a busy three weeks for a ten-person team. The reproducer was a 280-second wall clock per build: about 230 seconds to compile and 50 seconds to run the export and measure it. A linear scan, worst case, would have been 4,096 builds at 280 seconds each, which is 318 hours of serial compute — call it thirteen days. Nobody does that; in practice people test a dozen commits, guess, and ship a maybe-fix.

The bisect run took the base-two log of 4,096, which is exactly 12 builds. Twelve builds at 280 seconds is 3,360 seconds — 56 minutes. I started it, went to a meeting, and came back to a finished bisect pointing at commit `9f8e7d6`, "Tune default export batch size for memory," a one-line change from `BATCH_SIZE = 1000` to `BATCH_SIZE = 1`. The author had been chasing a memory spike in a different code path and dropped the batch size globally; it fixed their memory issue and multiplied the export's database round trips by a thousand. The before-and-after, once we knew where to look, was trivial to prove: revert that one line on a branch and export latency went from 90 seconds back to 4. Twelve builds, 56 minutes, one line. That is the number I bring to every "let's just read the code and figure it out" conversation. Reading the code would never have found it, because the export code was correct — the *default it depended on* had moved.

## 5. Bisecting beyond code: the insight that makes this a general tool

Here is the realization that turns bisection from "a git command" into a way of thinking. `git bisect` is binary search over commits, but **nothing about binary search cares that the axis is commits**. The algorithm needs exactly two things: an ordered axis, and a yes-or-no test that is monotonic along it. Commits ordered by time, tested by "is the bug present?", happen to satisfy both. But so do many other axes you encounter every day. Once you see this, you start binary-searching things you used to grind through linearly.

![A seven row matrix mapping code, data, config, dependency versions, time, feature flags, and a service chain each onto a way to split the half and the resulting small probe count](/imgs/blogs/binary-search-your-bug-with-bisection-5.png)

The matrix above is the generalization, and it is worth internalizing every row. **Code commits** are the `git bisect` case: split the commit DAG, log-base-two of 4,096 is twelve builds. **Data rows**: a parser crashes on a 50,000-row file and you have no idea which record; delete half the rows, see if it still crashes, recurse into the crashing half — sixteen halvings because log-base-two of 50,000 is about sixteen. **Config flags**: a 1-million-line generated config makes the service misbehave and you cannot read it all; comment out half, test, recurse — about twenty checks. **Dependency versions**: a bug appeared somewhere across 100 releases of a library; binary-search the version you pin in the lockfile, reinstall, test — seven installs. **Time windows**: a metric broke sometime in the last 256 hours of deploys and you do not know when; bisect the deploy timeline or the log window — eight probes. **Feature flags**: 32 flags are on and one combination triggers the bug; toggle half off, test, recurse — five toggles. **Service chain**: a request flows through 10 microservices and one is at fault; short-circuit half the chain with a stub, test, recurse — four probes. Every one of these is the same algorithm wearing a different costume.

The mental shift is to stop asking "where is the bug?" and start asking "what axis can I split in half, and what is my yes-or-no test?" The moment you can phrase your problem as "I have a range, and I can tell whether the bug is present at any point in it," you have a twelve-step solution to a four-thousand-suspect problem. Let me make three of these rows concrete, because reading about them is one thing and the code is another.

### Bisecting the data

Suppose a batch importer crashes on a 50,000-row CSV with an unhelpful `UnicodeDecodeError` deep in a third-party parser, and the stack trace does not tell you which row. You could add print statements and watch them scroll, but that is linear. Instead, bisect the file. The test is "does the importer crash on this slice of rows, yes or no?", the axis is row number, and it is monotonic in the sense that the slice containing the bad row crashes and the slice without it does not. Here is a bash loop that bisects the dataset by halving:

```bash
#!/usr/bin/env bash
# bisect-data.sh — find the single row that crashes the importer in a 50000-row file.
set -u
INPUT="data.csv"
HEADER=$(head -n 1 "$INPUT")
lo=2                    # first data row (line 1 is the header)
hi=$(wc -l < "$INPUT")  # last line number

while (( lo < hi )); do
    mid=$(( (lo + hi) / 2 ))
    # Build a candidate file: header + rows [lo..mid].
    { echo "$HEADER"; sed -n "${lo},${mid}p" "$INPUT"; } > /tmp/slice.csv

    if ./import /tmp/slice.csv > /dev/null 2>&1; then
        # Lower half is clean — the bad row must be in the upper half.
        lo=$(( mid + 1 ))
        echo "rows ${lo}..${mid} ok, searching ${lo}..${hi}"
    else
        # This half crashes — recurse into it.
        hi=$mid
        echo "rows up to ${mid} crash, searching ${lo}..${hi}"
    fi
done

echo "first bad row is line ${lo}:"
sed -n "${lo}p" "$INPUT"
```

This converges in about sixteen iterations and prints the exact offending line, which you can then inspect to see the malformed byte sequence. No print-statement archaeology, no reading 50,000 rows — sixteen runs of the importer and you have the record.

![A four row grid showing a fifty thousand row file split into halves where the second half crashes, then recursing into the crashing half twice more until a single malformed record at row forty one thousand is isolated](/imgs/blogs/binary-search-your-bug-with-bisection-6.png)

The grid above traces the convergence: first the file splits into rows 1 to 25,000 (parses fine) and 25,001 to 50,000 (crashes), so you recurse into the upper half; that splits into 25,001 to 37,500 (fine) and 37,501 to 50,000 (crashes); and after sixteen total halvings the surviving range is one row — line 41,209, a record with a truncated multi-byte UTF-8 sequence the parser cannot decode. The same logarithmic collapse as the commit case, applied to rows instead of commits.

### Bisecting the config

A service reads a 4,000-line generated YAML config and behaves wrongly on startup, but you have no idea which setting is responsible. Bisect the config: comment out the bottom half of the settings (or, more safely, replace the file with a known-minimal config plus the top half of the suspect settings), restart, test. If the bug persists, the culprit is in the top half; if it vanishes, it is in the bottom half you removed. Recurse. A 4,000-line config falls in about twelve toggles, and a million-line one in about twenty. The trick that makes config bisection clean is to have a *known-good minimal baseline* and add suspect settings in halves, rather than removing settings from the full file — removing a setting can itself change behavior in ways unrelated to the bug, whereas adding to a minimal baseline keeps the test honest.

### Bisecting time and deploys

Sometimes the axis is *time itself*. A success-rate metric was 99.9% on Monday and is 97% today, and you deploy a dozen times a day, so "what broke it?" is really "which deploy?" If your deploys are tagged in git or recorded with timestamps, you can bisect the deploy history exactly like commits. If you cannot redeploy old versions cheaply, you bisect the *observation* instead: pick a time in the middle of the degraded window, query the metric for the hour around it, and ask "was it already broken here?" Monotonic in time (once it broke, it stayed broken), so binary search over the time window finds the deploy boundary in log-base-two of the number of deploys. Eight deploys' worth of degraded window is three probes; 256 is eight. This is the same reason "when did the dashboard first show the spike?" is a binary-search question, not a scroll-through-the-logs question.

### Bisecting a service chain

A request traverses ten services — gateway, auth, three business services, a cache, two datastores, a search index, a notifier — and the response is wrong, but you do not know which hop corrupts it. You can bisect the *chain*: stub out the back half of the services so the request only goes halfway, and check whether the (partial) response is already wrong. If it is, the fault is in the front half; if the front half is clean, it is in the back half you stubbed. Four probes localize the bad service among ten. This is harder to automate than `git bisect run` because stubbing services is fiddly, but the *thinking* is identical, and it beats reading ten services' worth of code. When the chain is too coupled to stub cleanly, the same logic applies to **distributed tracing**: a trace already records each hop's timing and status, so you binary-search the span tree by asking "is the corruption present at this span?" For the cross-service mechanics of stubbing, correlation IDs, and trace propagation, see [observability with metrics, logs, and traces by design](/blog/software-development/system-design/observability-metrics-logs-traces-by-design); bisection is the search strategy you run *on top of* that instrumentation.

## 6. `git bisect skip` and the commits you cannot test

Real history is not uniformly testable. Somewhere in your range of 4,096 commits, a handful will not build — a mid-refactor commit where the code is in a broken intermediate state, a commit that depends on a migration that has not run, a commit from before a dependency existed in your lockfile. If `git bisect` checks one of these out and asks you "good or bad?", the honest answer is "I cannot tell, this commit does not even run." That is what `git bisect skip` is for.

When you skip, git removes that commit from consideration as a *test point* but keeps it in the range, and picks a nearby commit to test instead. In a `bisect run` script, you signal skip with **exit code 125** — which is exactly why the export script above exits 125 when `make build` fails. Git will try other commits near the un-buildable one until it can make progress. If a whole *region* is un-buildable, git may end up reporting a range of candidates rather than a single commit, because it genuinely cannot narrow further — and that honest "the first bad commit is one of these N" is far more useful than a false precise answer.

```bash
# Manual skip when you check out a commit that will not build or test:
$ make build
... compilation error ...
$ git bisect skip
Bisecting: 340 revisions left to test after this (roughly 9 steps)
[c3d4e5f] ...
```

A subtle and important point: `skip` is for "I cannot test this commit," not for "I do not feel like testing this commit." If you skip commits that are actually testable just to save time, you are throwing away the precision bisection gives you. Skip only the genuinely un-runnable ones. And if you find that *most* of your range does not build, that is a signal to bisect a different artifact — for example, bisect the merge commits only (which are usually buildable) with `git bisect skip` on everything else, or bisect a CI artifact rather than building from source.

## 7. The pitfalls: when bisection lies, and how to make it tell the truth

Bisection's one assumption — monotonicity, a single clean good-to-bad transition — is also its one weakness. When that assumption is violated, the midpoint test gives you a wrong answer, and a wrong answer early in a bisect sends the whole search down the wrong branch, so you confidently converge on an *innocent* commit. The dangerous part is that bisection does not warn you; it happily produces a precise, wrong result. So you have to know the violation modes and defend against each. The decision tree below is the field guide.

![A decision tree rooting at whether the range has one clean good to bad flip and branching to a flaky bug, an unbuildable commit, and a bad baseline each with its specific counter move](/imgs/blogs/binary-search-your-bug-with-bisection-7.png)

The tree above starts at the question you must ask before trusting any bisect: *does this range have one clean flip from good to bad?* If not, one of three things is usually wrong, and each has a counter-move.

**The bug is flaky (non-monotonic in the worst way).** The bug comes and goes — it fires on some runs and not others, regardless of commit. Now "the test passed at this commit" does not mean "the bug is absent here"; it might just mean this run got lucky. A single false-good early in the bisect throws away the half of the range that actually contains the culprit, and you converge on garbage. The counter-move is to run the test *many times* per step and only call the commit good if it passes *every* time — any single failure marks it bad. We will do the math on how many times in section 8. The key intuition: a flaky bug is monotonic in *probability* even if not in any single run, so you make each probe statistically sound by repeating it.

**A commit in the middle does not build.** Covered above — `git bisect skip`, exit 125. The danger if you *don't* skip is that you mark an un-buildable commit "bad" (because the test "failed"), which conflates "the bug is here" with "this commit is broken in an unrelated way," and again sends the search astray. Skip is the honest answer.

**The baseline was not actually good.** You told git `git bisect good v2.3`, but what if the bug was *already present* at `v2.3` and you just never noticed? Then there is no good commit in your range at all, the "good-to-bad transition" you are searching for does not exist inside it, and bisection will still dutifully report some commit as "first bad" — a lie, because the real first-bad is older than your baseline. The counter-move is to *test your good commit first*, by hand, before you trust the bisect: check out `v2.3`, build, run the repro, and confirm it really is good. If it is not, extend the range backward (pick an even older known-good point) and start over. A bad baseline is the most insidious failure because it produces a clean, confident, wrong answer with no symptom that anything went wrong.

There is a fourth pitfall the tree does not show because it is less a violation than a complication: **bisecting through a refactor that moved the bug.** Sometimes the "first bad commit" bisection reports is a refactor that *renamed or moved* the buggy code, not the commit that introduced the fault. The fault existed before, in a form your reproducer did not trigger, and the refactor changed the conditions just enough to expose it. Bisection is not wrong here — that refactor genuinely is where the *observable* behavior flipped — but the *root cause* may be older. The tell is that the reported commit is a pure rename or move with no logic change. When you see that, treat the reported commit as "where it became visible," read its diff to understand what condition changed, and consider a second bisect on a reproducer tuned to the older form of the bug.

#### Worked example: a flaky regression that fooled the first bisect, then fell to a repeated-probe bisect

This one cost me an afternoon and is the reason I now repeat every flaky probe. A test, `test_checkout_concurrency`, started failing intermittently on CI — maybe one run in three — sometime in the last 800 commits. I bisected it the naive way: `git bisect run` with a script that ran the test once and exited on its result. Bisection converged in ten steps and confidently named a commit that touched the checkout *logging*. That made no sense — adding a log line does not cause a concurrency bug — and indeed, reverting it changed nothing. The bisect had lied, because somewhere in the middle a flaky commit happened to *pass* its single run, got marked good, and the half of history containing the real culprit was discarded.

The fix was to make each probe statistically sound. I rewrote the script to run the test 20 times per commit and exit bad on *any* failure, good only if all 20 passed (the script is in the next section). With a true failure rate around one in three, the chance that a genuinely-bad commit passes 20 clean runs is roughly two-thirds to the twentieth power — about one in fifty thousand — so a "good" verdict became trustworthy. That bisect ran longer, of course: 20 test runs per step instead of one, ten steps, so 200 test runs at about eight seconds each, around 27 minutes. It landed on a completely different commit — one that had removed a lock around the inventory-decrement, introducing a genuine race in the concurrent checkout path. Reverting *that* made the test pass 2,000 times in a row. The first bisect was fast and wrong; the second was slower and right. The lesson is permanent: for a flaky bug, a single run per probe is not a measurement, it is a coin flip, and bisection built on coin flips converges on noise.

## 8. Saving, replaying, and steering a bisect

A long bisect is a record of decisions, and git treats it as one you can save, share, and replay — which matters more than it sounds, because a bisect that takes an hour is something you do not want to redo from scratch if you fat-finger a `good` for a `bad` halfway through. Three commands turn bisection from a one-shot session into a reproducible, auditable process.

`git bisect log` prints the full history of the current bisect — every `good` and `bad` and `skip` you (or the run script) issued, in order. It is your lab notebook for the search:

```bash
$ git bisect log
# bad: [9a8b7c6] current HEAD, export slow
# good: [1122334] v2.3, export fast
git bisect start 'HEAD' 'v2.3'
# bad: [a1b2c3d] Refactor the export serializer
git bisect bad a1b2c3d
# good: [d4e5f6a] Bump the database driver to 5.2
git bisect good d4e5f6a
# ... and so on
```

Save that to a file and you can replay the entire search on another machine, or after a `git bisect reset`, with `git bisect replay`:

```bash
$ git bisect log > /tmp/bisect.log    # capture the session
$ git bisect reset                    # bail out for now
# ... later, or on a colleague's checkout ...
$ git bisect replay /tmp/bisect.log   # re-run every good/bad/skip to the same point
```

This is genuinely valuable in two situations. First, **recovering from a mistake**: if you realize at step 8 that you marked step 3 wrong (you said `good` but that commit was actually bad), you do not start over — you dump the log, hand-edit the offending line, and `replay` the corrected log, and git fast-forwards through all the decisions to the corrected state. Second, **handoff**: you can hand a colleague your partial bisect log so they continue the search where you left off, or attach it to a bug report so a maintainer can reproduce your localization exactly. The Linux kernel community asks for exactly this — "attach your `git bisect log`" — because the log *is* the reproducible artifact of the search.

There is also `git bisect visualize` (alias `git bisect view`), which opens `gitk` or your log viewer scoped to just the commits still under suspicion, so you can eyeball the remaining range — useful when you suspect a merge is muddying things or you want to read the commit messages of the survivors before the next probe. And a small but powerful steering tool: `git bisect start -- <pathspec>`. If you already suspect the regression is in one subsystem, you can restrict the bisect to commits that touched specific paths:

```bash
$ git bisect start -- src/export/    # only bisect commits that touched the export module
$ git bisect bad
$ git bisect good v2.3
```

Now git ignores commits that did not touch `src/export/` when picking midpoints, which can dramatically shrink the effective search and skip irrelevant noise — *provided* your suspicion is correct. The caution is the same as always: if the real cause is *outside* the path you scoped to (the slow export was caused by a config default in `src/config/`, not `src/export/`), the path-scoped bisect will confidently land on the wrong commit, because you told it to ignore the actual culprit. Scope the path only when you are confident the cause lives there; when in doubt, bisect the whole range — twelve steps is cheap.

#### Worked example: bisecting a 1.2-million-line config to one toggled flag in 21 checks

The export team had a second incident that the commit-bisect could not touch, because *no code changed* — the regression came from a generated configuration. A nightly job assembles the service config from a hundred upstream sources into a single 1.2-million-line file, and one morning the service started rejecting 8% of valid requests. Diffing yesterday's config against today's was useless: the diff was forty thousand lines of churn from re-ordered sources, none of it obviously the cause.

So I bisected the config. The axis was "lines of the config file," the test was "does the service reject valid requests after loading this config?", and it was monotonic in the sense that any config slice containing the bad setting reproduced the rejections. I started from a known-good minimal baseline config (the defaults, which I had verified did *not* reject), then added the suspect lines in halves: first the top 600,000 lines on top of the baseline. Service rejected requests — the bad setting was in the top half. I added the top 300,000 instead — clean. So the culprit was in lines 300,001 to 600,000; I added those. Rejected. And so on, halving each time. Log-base-two of 1.2 million is about 20.2, so it took 21 restart-and-test cycles, each about 40 seconds (restart plus a 200-request probe), totaling about 14 minutes. It landed on a single line: a feature gate, `strict_schema_validation`, that an upstream source had flipped from `false` to `true`, which started rejecting requests carrying a deprecated-but-still-valid field. Twenty-one checks, 14 minutes, one flipped boolean, in a file no human could read. A linear scan of 1.2 million lines is not a thing a person does; bisection made it a coffee break. The before-and-after was a one-line config revert and the rejection rate dropped from 8% back to 0%.

The general lesson from both worked examples: when *your* code did not change but the behavior did, the regression is in something you *depend on* — a config, a default, a dependency version, an environment variable — and every one of those is a binary-searchable axis. Do not limit bisection to `git`; bisect whatever changed.

## 9. Bisecting a flaky bug: the statistics of how many runs per step

Let me make the flaky-bisect math precise, because "run it a bunch of times" deserves an actual number. Suppose a genuinely-bad commit reproduces the bug with probability `p` on any single run — say `p` equals one-third, so the test fails one run in three at a bad commit and never fails at a good commit. The danger is a **false good**: a bad commit whose single run happens to pass, fooling bisection into discarding the wrong half.

If you run the test once, the probability you get a false good at a bad commit is one minus `p`, which here is two-thirds — a 67% chance of being misled at every bad commit you probe. That is catastrophic for bisection. If you run the test `n` times and mark the commit bad if *any* run fails, then a false good requires *all* `n` runs to pass, which happens with probability `(1 - p)` to the power `n`. With `p` equal to one-third and `n` equal to 20, that is two-thirds to the twentieth, about 0.0003 — roughly one in fifty thousand. Across a 10-step bisect with, say, 5 bad commits probed, your chance of *any* false good is about 5 times 0.0003, well under one percent. Now the bisect is trustworthy.

![A vertical stack showing a flaky bug that fires a third of the time, why one run per step misses it, looping the repro twenty times per commit, and the resulting false good odds of one in fifty thousand](/imgs/blogs/binary-search-your-bug-with-bisection-8.png)

The stack above is the whole argument: a bug that fires about a third of the time, one run per step that misses it two-thirds of the time, looping the repro 20 times per commit and marking bad on any failure, with the resulting false-good odds of two-thirds to the twentieth, about one in fifty thousand. The general rule: pick `n` so that `(1 - p)` to the `n` is comfortably below your tolerance — if you can estimate `p`, solve for `n` as the logarithm of your tolerance divided by the logarithm of `(1 - p)`. If you cannot estimate `p`, measure it: run the test 100 times at the known-bad `HEAD`, count failures, and that fraction is your `p`. Here is the repeated-probe `bisect run` script:

```bash
#!/usr/bin/env bash
# bisect-flaky.sh — run the flaky test up to 20 times; bad on ANY failure, good only if all pass.
set -u
RUNS=20

# Build first; un-buildable commits are skipped, not judged.
if ! make build > /tmp/build.log 2>&1; then
    exit 125
fi

for i in $(seq 1 "$RUNS"); do
    if ! ./run-test test_checkout_concurrency > /dev/null 2>&1; then
        echo "FAILED on run $i of $RUNS at $(git rev-parse --short HEAD) — bad"
        exit 1          # any single failure ⇒ the bug is present ⇒ bad
    fi
done

echo "passed all $RUNS runs at $(git rev-parse --short HEAD) — good"
exit 0                  # all clean ⇒ good (with ~1-in-50000 false-good risk)
```

Two refinements worth knowing. First, you can *short-circuit*: the script already exits bad on the first failure, so a bad commit usually fails fast (on average after about `1/p` runs, three runs here) rather than always doing all 20 — only *good* commits pay the full 20-run cost. That makes the flaky bisect much cheaper than "20 runs everywhere" suggests. Second, if even 20 runs is not enough because `p` is tiny (a one-in-a-thousand bug), you may need to first make the bug *more reproducible* before bisecting — increase concurrency, add load, run under ThreadSanitizer to surface the race deterministically — which is the reproduce stage of the loop. You cannot reliably bisect what you cannot reliably reproduce; [reproduce it first or you are not debugging](/blog/software-development/debugging/reproduce-it-first-or-youre-not-debugging) is the prerequisite for this whole technique. Bisection searches; reproduction is what makes each probe answerable.

## 10. Bisecting inside a single function: short-circuit the halves

Bisection scales all the way down to a single function. Suppose a 400-line function produces a wrong result and you do not want to step through it line by line in a debugger. Bisect the *code path*: insert an early `return` (or a `raise`/`panic`) at the halfway point with the partial state, run it, and check whether the intermediate result is already wrong at the midpoint. If the value is already corrupt at line 200, the bug is in the first half; if it is still correct at line 200 but wrong at the end, the bug is in the second half. Recurse by moving the early return. A 400-line function falls in about nine probes — log-base-two of 400 — instead of 400 lines of reading.

```python
def compute_invoice(order):
    subtotal = sum_line_items(order)        # lines 1-100
    taxed = apply_taxes(subtotal, order)    # lines 101-200
    # --- BISECTION PROBE: short-circuit here and inspect the midpoint state ---
    print(f"PROBE midpoint: subtotal={subtotal} taxed={taxed}")
    raise SystemExit  # comment out once this half is cleared
    discounted = apply_discounts(taxed, order)  # lines 201-300
    total = round_to_cents(discounted)          # lines 301-400
    return total
```

If `taxed` is already wrong at the probe, the bug is in `sum_line_items` or `apply_taxes` and you move the probe to line 100. If `taxed` is right, the bug is downstream and you move the probe to line 300. This is the same algorithm as `git bisect`, run with your hands on the code instead of on commits — and it is often faster than setting up a debugger for a function you can just edit. It composes with the hypothesis-driven loop: each probe is one falsifiable claim ("the value is correct up to here") and one cheap experiment (run with the early return), exactly the discipline from [hypothesize and falsify](/blog/software-development/debugging/hypothesize-and-falsify-not-stare-and-hope), applied to a position in a function.

The one caveat for in-function bisection is **side effects across the cut**. If `apply_taxes` not only computes a value but also writes to a database or mutates shared state, an early `return` skips the writes that later lines depend on, and you may produce a *different* failure than the real one — a false signal. The clean version of the technique inspects state at the midpoint *without* short-circuiting the rest, by logging the intermediate value and letting the function run to completion: you check whether `taxed` was already wrong at line 200 in the logs, regardless of what happens after. Reserve the hard `raise SystemExit` for pure, side-effect-free computations where stopping early changes nothing but speed. For impure code, the probe is a log line, not an early return — same bisection logic, no collateral damage.

### Stress-testing your bisect: what if it only reproduces under load, in release, or on one host?

The hardest bisects are the ones where the bug only shows up under conditions you cannot easily recreate at every probe, and this is where the reproduce stage and the bisect stage fuse. Walk through the awkward cases. *What if the bug only reproduces under load?* Then a bisect probe that runs the code once in isolation will report every commit as good, and bisection converges on nothing. The fix is to fold the load into the probe: your `bisect run` script must itself generate the load — spin up the concurrency, fire the request volume, run `stress-ng` — so that each probe reproduces the conditions the bug needs. The probe gets slower and more expensive, but it stays *honest*, which is the only thing that matters.

*What if it only reproduces in a release build at `-O2` but not in the debug build you would normally bisect?* Then bisect the release build — your script must compile with the same optimization flags and run the same binary that exhibits the bug. A bug that is an optimizer-exposed undefined behavior or a timing-sensitive race will simply vanish at `-O0`, and a bisect on the debug build will find a fictional transition or none at all. Match the build configuration to the one that breaks.

*What if it only reproduces on one host, or one architecture, or after six hours of uptime?* The host-specific case means you must run the bisect *on that host* (or an identical one), because the difference is in the environment, not just the code — and if the difference turns out to be the environment entirely, you bisect the *environment* (which package, which kernel, which config) rather than the code. The six-hours-of-uptime case is the cruelest: a probe that needs six hours to manifest makes even a twelve-step bisect a multi-day affair. Here the right move is to *amplify* the bug before bisecting — find what accumulates over six hours (a slow leak, a counter wrapping, a cache filling) and accelerate it (shrink the buffer, raise the rate, fast-forward the clock) so the probe reproduces in minutes. You spend effort making the reproducer fast precisely because the bisect multiplies that cost by a dozen. The principle is constant: bisection is only as trustworthy as its probe, so invest in a probe that reliably reproduces the bug under the conditions it needs, and the logarithm takes care of the rest.

## 11. A comparison of what bisects well and what does not

Not every problem is a good fit for bisection, and a senior engineer reaches for it deliberately rather than reflexively. The table below is the trade-off space.

| Search axis | Yes/no test | Probes for N | Monotonic? | Gotcha |
| --- | --- | --- | --- | --- |
| Code commits | Build + run repro, exit code | log2(N), ~12 for 4096 | Usually yes for regressions | Flaky bug, bad baseline, refactor moved it |
| Dataset rows | Feed a slice, does it crash | log2(N), ~16 for 50k | Yes if one bad record | Multiple bad rows break it |
| Config lines | Toggle half, restart, test | log2(N), ~20 for 1M | Usually yes | Setting interactions (two flags) |
| Dependency versions | Pin version, reinstall, test | log2(N), ~7 for 100 | Yes if one bad release | API changes block install (skip) |
| Time / deploys | Query metric at midpoint | log2(N), ~8 for 256 | Yes if step change | Gradual drift is not a step |
| Feature flags | Toggle half on, test | log2(N), ~5 for 32 | Often no | Combinations, not single flag |
| Service chain | Stub back half, check response | log2(N), ~4 for 10 | Usually yes | Stateful coupling resists stubbing |
| Single function | Early-return at midpoint | log2(lines) | Yes for dataflow | Side effects across the cut |

The pattern in the "gotcha" column is consistent: bisection assumes a *single* responsible thing along a *monotonic* axis. When the cause is a *combination* (two feature flags that only break together, two malformed rows that interact), naive bisection struggles, because removing half might remove one of a pair and "fix" the bug without isolating the cause. The counter-move for combination bugs is a different search (a delta-debugging algorithm like `ddmin`, which minimizes a *failing set* rather than finding a single transition point — it repeatedly removes subsets and keeps the smallest still-failing configuration). But the overwhelming majority of real bugs are a single transition on a monotonic axis, which is exactly why plain bisection is so productive in practice.

The other axis of the trade-off is *cost per probe*. Bisection is worth setting up when each probe is expensive enough that you cannot afford to do N of them but cheap enough to automate. A five-minute build is the sweet spot: 4,096 of them is impossible, 12 is an hour, and a script can run them. If a probe takes a full day (a slow integration suite, a manual reproduction that needs a human), bisection still wins on *count* but each step hurts, so you invest in making the probe cheaper first — a faster reproducer, a smaller test — before you bisect. And if a probe is sub-second and you have only twenty candidates, just scan them linearly; the setup cost of a bisect is not worth it. Reach for bisection when N is large and the per-probe cost is automatable-but-not-trivial. For the broader question of building a fast, scriptable reproducer — the thing every probe depends on — see [reproduce it first or you are not debugging](/blog/software-development/debugging/reproduce-it-first-or-youre-not-debugging).

## 12. War story: bisecting real-world regressions

Bisection has a long track record on famous and infamous bugs, and a couple of real cases illustrate both its power and its limits.

**The Linux kernel and `git bisect`.** Bisection is not a niche trick; it is core to how the Linux kernel is maintained, and `git bisect` was built by and for the kernel community precisely because the kernel sees hundreds of commits per day from thousands of contributors. When a kernel regression is reported — "my laptop will not suspend after upgrading from 5.14 to 5.15" — the standard response from maintainers is "please `git bisect` it." A user with no knowledge of the kernel internals can build kernels at midpoints, test "does suspend work?", mark good or bad, and hand the maintainer a single offending commit. Tens of thousands of commits separate two kernel releases; bisection turns that into roughly fifteen builds. The kernel's `bisect run` scripts routinely localize regressions across an entire release cycle unattended overnight. This is the canonical proof that bisection scales to the largest real codebases, and it is why "did you bisect it?" is the first question a kernel maintainer asks.

**The performance regression that was a one-character change.** A widely-retold class of incident — and one I have lived a version of — is a service that gets mysteriously slower after a routine upgrade, and the cause turns out to be a single-character or single-default change buried in a dependency. Bisecting the dependency versions (not your own code, theirs) is the move: pin the library to the midpoint of the suspect version range in your lockfile, reinstall, benchmark, mark good or bad. Across 100 releases that is seven installs, and it routinely lands on a release where someone changed a default connection-pool size, flipped a cache from on to off, or changed an algorithm's default from one variant to a slower-but-more-correct one. You would never find it by reading your own code, because your code did not change — you bisected *their* releases through your lockfile. For the git-history side of this — recovering, inspecting, and reasoning about exactly which version of what you are running — [using git like a senior, a workflow and troubleshooting playbook](/blog/software-development/version-control/using-git-like-senior-workflow-troubleshooting-playbook) and [git like a pro, the object model, workflows, and recovery](/blog/software-development/version-control/git-like-a-pro-object-model-workflows-and-recovery) are the deeper references on the version-control mechanics that bisection rides on.

**The flaky test that bisection found by repeating the probe.** The concurrency case from section 7 is the realistic-but-illustrative archetype of the *limit* of naive bisection. A real, documented pattern in large CI systems: an intermittent failure gets bisected once, lands on an innocent commit (often a logging or comment change that happened to be a midpoint that got lucky), the team reverts it, nothing improves, and trust in bisection erodes — "bisect gave us the wrong answer." It did not; the *single-run probe* gave the wrong answer, and bisection faithfully searched on noisy data. The teams that get value from bisecting flaky tests are the ones that repeat each probe enough times to make the verdict statistically sound, exactly the `bisect-flaky.sh` pattern above. The fix is not to distrust bisection; it is to give it a trustworthy test.

A note on accuracy: the Linux kernel's use of `git bisect` is well-documented and concrete. The performance-regression and flaky-test cases above are realistic composites of patterns I have seen repeatedly rather than a specific named company's published postmortem; I present them as illustrative of the technique, not as documented incidents. The arithmetic in all of them — log-base-two of the range — is exact and is the part you should trust.

## 13. How to reach for this (and when not to)

Bisection is so cheap and so powerful that the main failure mode is *not reaching for it* when you should. Here is the decisive guidance.

**Reach for bisection when** you have a regression with a known-good and known-bad point, the suspect range is large (dozens to thousands), each probe is automatable, and the property is plausibly monotonic. That covers a huge fraction of "it used to work and now it does not" bugs, which is most of the bugs that reach production. The moment you can say "this was fine at commit X and broken at commit Y," you should be typing `git bisect start` before you finish reading the code. If you can write a script that detects the bug from an exit code, use `git bisect run` and walk away — that is the highest-leverage twenty minutes in debugging.

**Reach past code into other axes** when the bug is not in *your* commits: bisect the data (which row), the config (which setting), the dependency versions (which release), the time (which deploy), the service chain (which hop). The generalization in section 5 is the whole point — train yourself to ask "what axis can I split?" reflexively.

**Do not bother bisecting when** the range is tiny (you have ten commits and a sub-second test — just scan them), when you have *no* known-good point (find one first, or you are bisecting a range with no transition in it), or when the bug is a *combination* rather than a single cause (use delta-debugging / `ddmin` instead, which minimizes a failing set rather than locating one transition).

**Do not trust a bisect result you have not sanity-checked.** Always verify your baseline is genuinely good before you start. Always repeat flaky probes enough times to make "good" mean good. Always read the diff of the reported first-bad commit and confirm it *plausibly causes* the bug — if it is a pure rename or a comment change, you have likely found "where it became visible," not the root cause, and you should look at what condition that commit changed. A bisect produces a precise answer with total confidence even when its assumptions are violated, so the precision is not evidence of correctness. The sanity check is.

**Do not waste an expensive probe.** If each build takes an hour, do not bisect 4,000 commits naively — that is still 12 hours. Invest in a cheaper reproducer first (a smaller test that triggers the same bug, a faster build target, a cached dependency layer), then bisect the cheap probe. Bisection's step count is fixed by the math; your only lever is the per-step cost, so spend your effort there.

The throughline of this whole series is *stop guessing and turn the search into a binary partition*. Bisection is that principle in its purest, most mechanical form: you do not need a theory of the bug, you do not need to understand the code, you need only a yes-or-no test and a range, and the logarithm does the rest. It is the bridge between the hypothesis-driven thinking of the [scientific method of debugging](/blog/software-development/debugging/stop-guessing-the-scientific-method-of-debugging) and the brute, reliable arithmetic of binary search — twelve probes for four thousand suspects, every time.

## Key takeaways

- **Bisection trades linear search for logarithmic search.** A bug hiding in N suspects falls in log-base-two of N probes: 12 for 4,096, 16 for 50,000, 20 for a million. That is a change in growth class, not a constant-factor win.
- **`git bisect start / bad / good` localizes a regression to one commit.** Git checks out the midpoint, you mark good or bad, and every answer halves the range. `git bisect reset` returns you to the present when done.
- **`git bisect run <script>` is the killer feature.** Write a script that exits 0 for good, 1 to 124 for bad, 125 for skip, and git finds the first bad commit unattended across thousands of commits in minutes.
- **Anything you can split in half and test is binary-searchable.** Bisect the data (which row), the config (which line), the dependency versions (which release), the time (which deploy), the feature flags (which combination region), the call stack (which half of a function), the service chain (which hop). The algorithm does not care what the axis is.
- **Bisection assumes monotonicity: one clean good-to-bad transition.** Every failure mode is a violation of that assumption, and every fix restores it.
- **For flaky bugs, repeat each probe.** A single run per step is a coin flip; run the repro `n` times and mark bad on any failure, so a false good costs `(1-p)` to the `n` — about one in fifty thousand at 20 runs of a one-in-three bug.
- **Use `skip` (exit 125) for un-buildable commits**, never for ones you simply do not want to test. Skipping testable commits throws away precision.
- **Always verify your baseline is truly good and read the first-bad diff.** A bad baseline or a refactor that merely *exposed* an older bug produces a confident, wrong answer with no warning.
- **Reach for bisection the moment you can say "fine at X, broken at Y."** It needs no theory of the bug — just a range and a yes-or-no test — which makes it the highest-leverage move in the whole debugging loop.

## Further reading

- [Stop guessing: the scientific method of debugging](/blog/software-development/debugging/stop-guessing-the-scientific-method-of-debugging) — the series intro that frames the observe, reproduce, hypothesize, bisect, fix, prevent loop that bisection is the engine of.
- [Hypothesize and falsify, not stare and hope](/blog/software-development/debugging/hypothesize-and-falsify-not-stare-and-hope) — the hypothesis-driven discipline that makes each bisection probe a real, falsifiable experiment.
- [Reproduce it first or you are not debugging](/blog/software-development/debugging/reproduce-it-first-or-youre-not-debugging) — the prerequisite for bisection, because every probe depends on a fast, scriptable reproducer.
- [Using git like a senior: a workflow and troubleshooting playbook](/blog/software-development/version-control/using-git-like-senior-workflow-troubleshooting-playbook) — deeper git workflow context for the version-control side of bisecting commits and lockfiles.
- [Git like a pro: the object model, workflows, and recovery](/blog/software-development/version-control/git-like-a-pro-object-model-workflows-and-recovery) — the commit DAG and object model that `git bisect` searches over.
- [Observability with metrics, logs, and traces by design](/blog/software-development/system-design/observability-metrics-logs-traces-by-design) — the instrumentation that makes bisecting a service chain or a deploy timeline possible.
- The official `git bisect` documentation (`git help bisect`) — the authoritative reference for `start`, `good`, `bad`, `skip`, `run`, `reset`, and `log`/`replay`.
- *Why Programs Fail* by Andreas Zeller — the canonical academic treatment of systematic debugging, including delta debugging (`ddmin`) for the combination-bug case that plain bisection does not handle.
