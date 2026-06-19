---
title: "Hypothesize and Falsify, Not Stare and Hope: The Hypothesis-Driven Core of Debugging"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Learn to turn a symptom into a falsifiable hypothesis, design the one experiment that cuts your search space in half, and beat the cognitive traps that keep you staring at code that is already lying to you."
tags:
  [
    "debugging",
    "software-engineering",
    "scientific-method",
    "hypothesis-driven",
    "root-cause-analysis",
    "cognitive-bias",
    "falsification",
    "troubleshooting",
  ]
category: "software-development"
subcategory: "Debugging"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/hypothesize-and-falsify-not-stare-and-hope-1.png"
---

Here is a scene I have watched a hundred times, and lived through more than I would like to admit. An engineer has a bug. The total on the invoice is one cent too high. They open the file where totals are computed. They read it. They scroll up. They read the function above it. They scroll down. They read the loop. They lean back. They read it again. Forty minutes pass. Nothing has changed on the screen and nothing has changed in their head, except a growing certainty that the code "looks right," which is the single most dangerous sentence in our craft. Eventually they change something — maybe they add a `round()`, maybe they `+ 1` somewhere, maybe they delete a line and put it back — run it once, and either the cent is gone (relief, ship it, never understand why) or it is not (despair, scroll up, read it again).

That is **stare and hope**. It is the default mode of the human mind facing a broken machine, and it is almost entirely useless. The code is not going to confess by being looked at harder. Worse, staring actively reinforces the wrong theory: every time your eyes pass over the line you already decided was innocent, you re-convince yourself it is innocent, and the bug — which is almost always living inside exactly the thing you are sure about — gets one more pass of protection. You can stare at correct-looking code for an hour while the actual fault sits two function calls away in a config value you never printed.

The alternative is a discipline, and it is the single biggest thing that separates a senior engineer from a junior who is flailing. It is not that the senior knows the codebase better (often they do not — watch a good debugger drop into a foreign codebase and still find the bug in twenty minutes). It is that the senior runs a **loop**: turn the symptom into a precise, falsifiable hypothesis; predict what you would observe if the hypothesis were true; run the cheapest experiment that would produce or destroy that observation; look at the result; and update — kill the theory or keep it and narrow. Each turn of that loop, ideally, throws away half of everything it could have been. This is the scientific method, the actual one from a philosophy-of-science class, applied to a stack trace. The figure below is the whole post in one picture.

![A vertical stack showing the falsification loop from a symptom of a wrong total down through hypothesize, predict, experiment, observe, and update to a kill or keep decision](/imgs/blogs/hypothesize-and-falsify-not-stare-and-hope-1.png)

By the end of this post you will be able to do five concrete things. You will take any symptom and write it down as a hypothesis that predicts an observation, so that one experiment can confirm or kill it. You will design the *cheapest discriminating experiment* — the single test whose result differs between your two leading theories, so you learn the most per second spent. You will recognize, by name, the cognitive traps that make smart people waste afternoons (confirmation bias, anchoring, the "it can't be that" trap, the streetlight effect, sunk cost, correlation-versus-causation) and you will have a counter-discipline for each. You will run an **assumption audit** — listing everything you believe is true and testing the cheap ones first, because the bug lives in the belief you never checked. And you will keep a debugging log, a lab notebook, so that on a six-hour bug you never re-test a dead theory or lose the thread. This is the hypothesize stage of the series' master loop — observe, reproduce, hypothesize, bisect, fix, prevent — and it is the stage where most debugging sessions are won or lost. If you have not read the series intro, [the scientific method of debugging](/blog/software-development/debugging/stop-guessing-the-scientific-method-of-debugging) sets up the whole loop; this post is the engine inside it.

## 1. The difference between staring and hypothesizing

Let me be precise about what is wrong with staring, because "just think harder" is not actually the lesson — sometimes thinking is exactly right, and I will defend it later. The problem with staring is that it produces no *information*. Information, in the strict sense that matters for debugging, is reduction of uncertainty. Before you act, the bug could be in any of N places — N lines, N functions, N services, N config values, N commits since it last worked. Staring at code does not change N. You end the forty minutes with the same N you started with, minus a false confidence that some of them are innocent. An experiment, by contrast, partitions N: after you run it, the bug is either in the "yes" half or the "no" half, and you have thrown the other half away. That is the entire game. Debugging is a search, and every action either narrows the search or it does not.

![A two-column before-and-after contrast of the stare-and-hope workflow against the hypothesize-and-falsify workflow showing the search space staying full versus collapsing](/imgs/blogs/hypothesize-and-falsify-not-stare-and-hope-2.png)

The before-and-after above is the mindset shift. On the left, three steps that all leave the search space full: re-read, random fix, still broken. On the right, three steps that each shrink it: one claim that predicts an observation, one experiment that cuts the space in half, root cause in roughly log-base-2 of N steps. The number is not decoration. If you genuinely halve the space each experiment, then a bug that could be in 4,096 places falls in 12 experiments, because two to the twelfth is 4,096. That is the same math as [binary search and bisection](/blog/software-development/debugging/binary-search-your-bug-with-bisection), and it is not a coincidence — bisection is hypothesis-driven debugging where the hypothesis is always "the bug is in this half of the range." Everything in this post is bisection generalized from a range of commits to the full space of possible explanations: code, data, config, time, the call stack, the environment.

Here is the philosophical engine, because it genuinely helps to know it. Karl Popper's insight about science was that you can never *prove* a theory true by accumulating confirming evidence — a million white swans do not prove all swans are white — but you can *prove a theory false* with a single counterexample. One black swan and the theory is dead. The asymmetry is total: confirmation is weak and endless, falsification is strong and final. Applied to your bug: do not look for evidence that your favorite theory is right (you will always find some, because confirmation bias makes the world look agreeable). Look for the experiment that would *kill* your theory if it is wrong. If you try hard to kill it and it survives, *now* you can trust it. A hypothesis that has survived a real attempt at falsification is worth a hundred that merely "seem consistent with what I'm seeing."

So a good hypothesis has a specific shape. It is not "something's wrong with the cache." That is a feeling, not a hypothesis — there is no experiment that confirms or kills it. A hypothesis is "the cache is returning stale data for this key," and crucially it comes with a *prediction*: "if that is true, then when I bypass the cache and read straight from the source, the bug will disappear." Now you have something to do. You add a flag, route around the cache, run the request, and look. Bug gone: hypothesis strongly confirmed, you have localized the fault to the cache layer and cut the entire rest of the system out of the search. Bug present: hypothesis dead, the cache is innocent, and — this is the part people miss — *that is a great result too*, because you just removed the whole caching subsystem from suspicion in one cheap experiment. A killed hypothesis is not a failure. It is a successful experiment with a negative result, and negative results narrow the space exactly as well as positive ones.

## 2. Turning a symptom into a falsifiable hypothesis

The hardest part for most people is the translation step: symptom to hypothesis. The symptom is what the world hands you — "the invoice total is one cent high," "the page is blank on Safari," "the job hangs after six hours," "p99 latency tripled last Tuesday." None of those is testable as written. The skill is to propose a *mechanism* — a specific, concrete story about how the symptom is produced — and then to state the observation that mechanism predicts.

Take the one-cent invoice. The symptom is "total is wrong by one cent in the high direction, sometimes." Notice I already enriched it: "in the high direction" and "sometimes" are data, and good hypotheses respect the data. A mechanism that predicts a *random* error in *both* directions is already inconsistent with "always high." Now propose mechanisms:

- **H1, rounding.** Somewhere we round each line item and the rounding mode is half-up where it should be banker's rounding, or we round at the wrong stage (per-line instead of on the sum). Prediction: the error should correlate with the number of line items that have a fractional half-cent, and it should be at most a cent or two. It should *not* appear on invoices with whole-dollar lines.
- **H2, double-counting.** One line item is being added to the sum twice — an off-by-one in a loop, a duplicated entry in the source list, a join that fans out. Prediction: the error should equal exactly the value of one line item, which on a one-cent discrepancy means one of the lines is worth one cent, *or* the doubled item is small. The count of items summed should be one more than the count displayed.

Both fit the symptom. That is the normal situation — you almost never start with one theory, you start with two or three that all explain what you see. The job is not to argue about which is more likely in the abstract (that is anchoring waiting to happen). The job is to find the experiment whose result *differs* between them.

![A branching graph in which a wrong-total symptom splits into a rounding hypothesis and a double-counting hypothesis that both feed one discriminating print experiment whose two outcomes kill one theory each](/imgs/blogs/hypothesize-and-falsify-not-stare-and-hope-3.png)

The graph above is the move. The symptom fans out to H1 and H2. Both point at *one* experiment: print the number of items summed and the raw subtotal before any rounding. That single print discriminates. If the count of summed items equals the count on the invoice and the subtotal is already one cent high before rounding, double-counting is impossible (count matches) and rounding is implicated. If the count is one greater than displayed, double-counting is confirmed and you do not care about rounding at all. One print, two theories, exactly one survives. We will write that print as real code in the worked example below.

The general recipe for the translation step is worth memorizing:

1. **Restate the symptom with every constraint you actually have.** Always high, not sometimes low. Only on Safari, not Chrome. Only after six hours, not at startup. Only under concurrent load, not single-threaded. Every adjective is a constraint that kills hypotheses for free.
2. **Propose two or three mechanisms** that would produce exactly that constrained symptom. If a mechanism predicts a symptom you do not see (errors in both directions, failures on Chrome too), it is already weakened or dead — that is falsification before you even run anything.
3. **For each mechanism, write the prediction** in the form "if this is the cause, then when I do X, I will observe Y." If you cannot write that sentence, your hypothesis is too vague to test; sharpen it until you can.
4. **Pick the experiment that splits the mechanisms** — the one observation Y that comes out differently depending on which mechanism is true.

That fourth step is where the leverage is, and it deserves its own section.

## 3. Designing the cheapest discriminating experiment

Not all experiments are equal. Some cost a five-second print; some cost spinning up a load test that takes an hour to set up and twenty minutes to run. Some rule out one suspect; some rule out half the system. The senior move is to maximize **information gain per unit of cost** — to reach first for the experiment that is both cheap and discriminating.

Think about it as a partition. Before the experiment, your set of live hypotheses is some collection. A *discriminating* experiment is one whose outcome is different across that collection — some hypotheses predict outcome A, others predict outcome B. After you observe A, all the B-predicting hypotheses are dead. The best experiment is the one that splits your live hypotheses as evenly as possible (so whichever way it goes, you kill about half) and costs the least to run. An experiment that all your hypotheses predict the same outcome for is *worthless* no matter how cheap — you learn nothing, because every theory survives.

This is exactly Shannon information and it is exactly bisection. If you have 16 equally-likely suspects and you can run a test that asks "is the culprit in this set of 8?", a yes-or-no answer eliminates 8 either way, and you find the culprit in 4 tests because 2 to the 4th is 16. If instead you test suspects one at a time — "is it suspect #1? is it suspect #2?" — you might need 15 tests. Same suspects, same culprit, four times the work, because the one-at-a-time test only discriminates a single suspect from the rest while the halving test discriminates eight from eight.

![A three-by-three grid illustrating how each yes-or-no experiment halves the suspect set from sixteen down to one across four tests like a binary search](/imgs/blogs/hypothesize-and-falsify-not-stare-and-hope-7.png)

The grid above walks that halving from 16 suspects to 1 in four tests. Burn this into your reflexes: when you are about to run an experiment, ask "whichever way this comes out, how much of the space does it eliminate?" If the answer is "a tiny sliver," go design a better experiment first. The classic better-experiment in code is the **midpoint probe**: do not check whether the bug is at the very start or the very end of a pipeline; check whether the data is already corrupt *halfway through*. If it is corrupt at the midpoint, the bug is in the first half and you have eliminated the entire second half in one read. If it is clean at the midpoint, the bug is in the second half. This is why a single well-placed log line in the middle of a transformation chain is worth ten log lines scattered randomly — the middle one halves, the scattered ones do not.

Here is a comparison of experiment designs by their information yield, which I keep in my head as a hierarchy of what to reach for:

| Experiment design | What it discriminates | Cost | Information per run |
| --- | --- | --- | --- |
| Midpoint probe (is data corrupt halfway?) | first half vs second half of pipeline | one log line | high — halves the space |
| Bypass a subsystem (route around cache/queue) | this subsystem vs everything else | a flag, minutes | high — removes a whole layer |
| Toggle a feature flag on/off, A/B | this change is causal vs coincidence | one config flip | high — proves causation |
| Targeted assert on an invariant | invariant holds vs broken at this point | one line, fails fast | high — pins the exact point |
| Single-stepping in a debugger | line N vs line N+1 | minutes of attention | low per step — use to confirm, not to search |
| Random print everywhere | almost nothing distinguishable | scattered noise | very low — you drown in output |
| Re-reading the code | nothing | 40 minutes | zero — N unchanged |

Notice the bottom two are the default human behaviors and they are at the bottom for a reason. Notice also that single-stepping in a debugger is *low* information per step — that is not a knock on debuggers, they are superb tools, but you use them to confirm a localized hypothesis ("step into this call and watch the value go wrong"), not to search a large space one line at a time. Reach for the high-information experiments first; drop to fine-grained tools once the space is small.

#### Worked example: the conditional breakpoint that discriminated on iteration 3,847,221

A batch job processed about 5 million records and produced one corrupted output row, but a different row each run — not deterministic by index. Two hypotheses fit: H-A, a specific *record* is malformed and corrupts whatever processes it (in which case the bad index would be stable across runs, tied to the data); H-B, a shared mutable buffer is occasionally not reset between records, so the corruption is *timing- or order-dependent* and lands on a different row each run. These predict different things, and the cheapest discriminating experiment is a conditional breakpoint that fires only when the corruption-detecting invariant first breaks, then inspects whether the *current record* is malformed (H-A) or *clean but the buffer is dirty* (H-B).

The information-gain logic is the same halving as everywhere else, but applied to which-hypothesis rather than which-line. You do not want to single-step five million iterations; you want the debugger to stop *exactly* at the first violation and hand you the state. In `pdb` (or `gdb` with a condition) the conditional breakpoint is the instrument:

```python
import pdb

def process(records):
    buf = bytearray(64)  # shared buffer — suspect for H-B
    for i, rec in enumerate(records):
        fill_buffer(buf, rec)
        out = transform(buf)
        # Hypothesis-as-condition: stop ONLY at the first corrupted output.
        # checksum_ok is our invariant; when it breaks, freeze and inspect.
        if not checksum_ok(out, rec):
            # We are now AT the first violation. Two questions, one breakpoint:
            #   H-A: is THIS record malformed?      -> print rec, validate it
            #   H-B: is the record clean but buf dirty from the LAST record?
            pdb.set_trace()    # inspect: rec valid? buf == expected for rec?
        yield out
```

The breakpoint fired on iteration 3,847,221. At the freeze, the current record was *valid* — H-A predicted a malformed record and was therefore killed on the spot — and `buf` contained trailing bytes from record 3,847,220 that `fill_buffer` had not overwritten because that record was shorter. H-B confirmed: a short record left stale tail bytes in the shared buffer, and only when a short record was immediately followed by one that read past its own length did the corruption surface, which is why it was order-dependent and non-reproducible by index. One conditional breakpoint, set on the invariant rather than a line number, discriminated the two hypotheses with a single stop out of five million iterations. The proof: zero `pdb` value at run-time once the fix landed (clear `buf` per record), and the checksum invariant held across 50 full runs of 5 million records each — 0 corrupted rows out of 250 million, where before it was reliably 1 per run. The lesson is that a debugger becomes a *high-information* instrument the moment you drive it with a condition that encodes your hypothesis, instead of stepping blindly.

## 4. The assumption audit: bugs live in what you are sure of

There is a special class of hypothesis that people systematically fail to test, and it is the most productive class of all: the things you *assume*. Every debugging session rests on a stack of beliefs you are not even aware you hold. The input is valid. The function actually got called. The config you think is loaded is the config that loaded. The network is up. The deploy you are looking at is the deploy that is running. The branch you are testing is the branch that is checked out. The clock is correct. The two services agree on the schema. You believe each of these so firmly that you never write them down, which is precisely why the bug hides there. A bug in code you are actively suspicious of gets found fast, because you test it. A bug in a belief you never articulated survives forever, because you never test it. The bug is in the haystack you are sure is just hay.

The counter-discipline is the **assumption audit**. Before (or early in) a frustrating debug, stop and literally write a list titled "things I believe are true here." Force yourself to ten items. Then sort them by *cost to verify* and test the cheapest first. Most of them will be true — that is fine, each verified assumption is a successful experiment that narrows the space and, just as valuably, restores your confidence that you can build on it. But every few audits, item seven turns out to be false, and item seven is your bug, and you would never have looked because item seven was "the database connection string points at the database I think it does."

![A decision tree of an assumption audit branching from the question of what you assume into input validity, code execution, and environment, each with a cheap verifying test where the config check reveals a wrong database host](/imgs/blogs/hypothesize-and-falsify-not-stare-and-hope-5.png)

The tree above shows the audit's shape: from the root question "what do I assume is true?" you branch into the input being valid, the code actually running, and the environment being what you think — and under each you hang the *cheapest test* that verifies it. Notice the tests are tiny: a shape assert, a one-shot log that proves a function was entered, a print of the resolved config. The audit is not heavy. It is fast, deliberate verification of the things you skipped because they were "obviously" fine. In my experience the single highest-yield audit item across a career is "is the code that's running the code I'm reading?" — wrong branch, stale build, cached bytecode, a deploy that silently failed, a second instance of the process, a sidecar overriding env. More wasted hours hide there than anywhere else. We will do a full worked example of an assumption audit catching exactly this further down — the "it can't be the database" bug.

Here is the assumption audit as runnable Python, because the cheapest assumptions are the ones you can assert in one line. A targeted assert is a hypothesis written in code: it states an invariant you believe, and it falsifies that belief loudly the instant it is wrong, at the exact point it is wrong, instead of letting a corrupted value drift ten function calls downstream and crash somewhere innocent.

```python
import logging
import os

log = logging.getLogger("audit")

def compute_invoice_total(line_items, config):
    # Assumption 1: the input is actually a non-empty list of items.
    # If this fires, the bug is upstream in how line_items got built.
    assert isinstance(line_items, list), f"expected list, got {type(line_items)}"
    assert len(line_items) > 0, "line_items is empty — upstream produced nothing"

    # Assumption 2: this function actually ran for THIS request.
    # A one-shot log proves the call happened and shows what it saw.
    log.info("compute_invoice_total called: n_items=%d", len(line_items))

    # Assumption 3: the environment is the one I think it is.
    # Print the resolved config, not the config I believe is loaded.
    log.info("resolved config: db_host=%s currency=%s rounding=%s",
             config.get("db_host"), config.get("currency"), config.get("rounding"))

    # Assumption 4: every item has the field we are about to sum.
    for i, item in enumerate(line_items):
        assert "amount_cents" in item, f"item {i} missing amount_cents: {item!r}"

    subtotal = sum(item["amount_cents"] for item in line_items)

    # Assumption 5 (the invariant): we summed exactly the items we displayed.
    # This is the discriminating assert for the double-counting hypothesis.
    assert len(line_items) == len({id(x) for x in line_items}), \
        "duplicate item object in line_items — double-counting"

    return subtotal
```

Each assert is a falsifiable hypothesis. The first two state "the input is well-formed"; if either fires, you have falsified that belief and the bug moved upstream — the experiment narrowed the space to "how `line_items` was built." The logs prove the call happened and expose the *actual* config (assumption three is the one that catches wrong-host bugs). The last assert is the discriminating test for double-counting written directly into the path. Cheap, loud, and each one converts a silent assumption into a tested hypothesis. The reason this beats a debugger for the audit phase is that asserts run on every execution including in CI and, with care, in production — they keep verifying the assumption long after you have stopped looking, which means the *next* time the belief becomes false you find out immediately instead of forty minutes into a stare.

## 5. Binary thinking: phrasing every experiment as yes or no

A subtle skill underlies everything above: phrasing experiments so the answer is a clean yes or no. A yes/no question partitions the world into exactly two sets, which is what lets you halve the space. An open-ended question — "what is the value of this variable across all these iterations?" — does not partition cleanly; it hands you a pile of data you then have to interpret, and interpretation is where bias creeps back in. Whenever you can, convert "what is happening?" into "is it the case that X?" because the second form forces a partition and forces a prediction.

Concretely, replace "let me look at the cache" with "does bypassing the cache make the bug disappear — yes or no?" Replace "let me check the timing" with "if I add a 500ms sleep here, does the race stop reproducing — yes or no?" Replace "is the input weird?" with "if I assert the input is sorted, does the assert fire — yes or no?" Each yes/no version comes attached to a prediction and an experiment; each open-ended version comes attached to a staring session. The conversion is mechanical once you practice it: take your vague urge to investigate something, and ask "what is the binary question whose answer would actually move me?"

This is also how you make use of the powerful debugging primitive of **toggling**. If you suspect a particular change, line, flag, or dependency is responsible, the cleanest experiment is to turn it off and on. Off: does the bug appear? On: does the bug appear? If the bug tracks the toggle — present when off, absent when on, or vice versa — you have a causal link, not a correlation. If the bug ignores the toggle, that suspect is innocent regardless of how guilty it looked. Toggling forces the binary, and it is the antidote to the correlation-versus-causation trap we will dissect shortly. A feature flag is not just a release-management tool; it is a debugging instrument that lets you A/B a suspected cause in production with one config change and no deploy.

```python
# A feature flag used as a debugging instrument, not just for releases.
# We suspect the new "fast path" sum is producing the wrong total.
# Toggle it to A/B the suspected cause against the known-good slow path.

def invoice_total(items, flags):
    if flags.enabled("fast_sum_path"):       # suspect ON
        return fast_sum(items)
    return slow_sum_reference(items)          # known-good reference

# Experiment, phrased as a binary:
#   flag OFF -> is the total correct?  (expect: yes, every time)
#   flag ON  -> is the total correct?  (expect: if guilty, NO)
#
# Run each variant 10 times against the failing invoice:
#   OFF: 10/10 correct
#   ON:  0/10 correct   -> fast_sum is causal, slow_sum is the reference truth
# If instead BOTH are wrong, the fast path is innocent and the bug is shared
# upstream (e.g. the items list itself), which eliminates the whole sum layer.
```

The comment block at the bottom is the experiment design, and it is deliberately written before running anything: state the prediction for each toggle position, then run and compare. If OFF is correct and ON is wrong, you have proven causation. If *both* are wrong, you just falsified the entire "it's the new fast path" theory and eliminated the sum layer — a hugely valuable negative result that points you upstream to how `items` was built. Either way you have learned something definite, because you forced the binary.

Binary thinking also lets you turn an *external* observation into a falsification when you cannot touch the code at all — the production process you must not attach a debugger to, the closed-source binary, the third-party dependency. The binary question becomes "does the process make the syscall I predict, yes or no?" and the instrument is a tracer. Suppose your hypothesis is "the service is reading the wrong config file, which is why it points at the stale replica." The prediction is sharp and binary: if true, the process will `openat` a config path you did not expect. You do not need the source; you watch the syscalls:

```bash
# Hypothesis: the process opens a config file other than the one I edited.
# Predict: an openat() on an unexpected path. Watch only file opens, follow forks.
strace -f -e trace=openat -p "$(pgrep -f reportservice)" 2>&1 \
  | grep -E 'config|\.ya?ml|\.env'
# Observed:
#   openat(AT_FDCWD, "/etc/reportservice/config.yaml", ...) = 5   <- I edited this
#   openat(AT_FDCWD, "/opt/app/config.yaml",          ...) = 7   <- I did NOT
# -> hypothesis CONFIRMED: a second config exists and wins. The file I edited
#    is read first and then OVERRIDDEN by /opt/app/config.yaml. Binary answer: yes.
```

That `strace` is a yes/no experiment on a running production process with zero code changes and no debugger attached — a hard requirement when the process is handling live traffic. The prediction ("an unexpected `openat`") was written before the trace; the observation either contains the surprising path or it does not. It did, and the "I edited the config" assumption — a textbook audit item — was falsified in one command: a second config file existed and won. For a production process you must not pause, a passive tracer is the right tool precisely because it observes without perturbing control flow, unlike a breakpoint that freezes the process.

## 6. The cognitive traps that wreck debugging

Now we get to the part that is really about you, not the code. The reason staring persists, the reason people chase dead theories for hours, the reason a bug "can't be" where it is — these are not failures of intelligence. They are predictable failures of cognition, the same biases that show up in every domain where humans reason under uncertainty. The good news is that because they are predictable and named, each has a specific counter-discipline you can deliberately invoke.

![A five-row matrix mapping each debugging bias of confirmation, anchoring, it-cannot-be-that, streetlight, and sunk cost to what it does and the counter-discipline that defeats it](/imgs/blogs/hypothesize-and-falsify-not-stare-and-hope-4.png)

The matrix above is the cheat sheet; let me give each row its due.

**Confirmation bias** is the master trap. You form a theory, and then your perception bends to confirm it — you notice the evidence that fits and unconsciously discount the evidence that does not. You run a test, get an ambiguous result, and read it as supporting your theory because that is the theory you have. The counter-discipline is pure Popper: *try to falsify your own favorite theory, not confirm it.* Before you go looking for evidence that you are right, design the experiment that would prove you wrong, and run that one. If your theory survives a genuine attempt to kill it, trust it. If you only ever run experiments that could confirm it, you have learned nothing, because a theory that cannot fail cannot teach. Concretely: when you catch yourself thinking "this should confirm it," stop and ask "what result would *disprove* this, and what's the cheapest way to look for that result?"

**Anchoring** is fixation on the first idea that entered your head. The first hypothesis arrives with a strange gravity; every subsequent observation gets interpreted relative to it, and you stop generating alternatives. The counter-discipline is to *write down two or three rival hypotheses before you test any of them.* The mere act of forcing yourself to produce alternatives breaks the anchor — now your first idea is one of three, not the One True Theory, and you will instinctively look for the experiment that discriminates among them rather than the one that pampers your favorite. If you only have one hypothesis, you do not have a hypothesis, you have a hunch, and a hunch with no rival is an anchor.

**The "it can't be that" / "I already checked that" trap** is the most expensive one, and it is confirmation bias wearing a confident hat. There is a region of the system you have decided is fine — you wrote it, or you checked it once, or it is "obviously" correct, or it is a mature library that surely works. So you exclude it from the search without testing it. And the bug, with the dark humor the universe reserves for debugging, is very often *right there*, in the part you refused to suspect. The counter-discipline is brutal and simple: *re-verify the assumption you are most sure of, especially the one you "already checked."* When you hear yourself say "it can't be the database" or "I already verified the input," treat that sentence as a flashing arrow pointing at where to look next. Did you actually check it, or did you check it *once, an hour ago, in a different state, and then change five things since?* "I already checked that" almost always means "I checked that under conditions that no longer hold." Check it again, now, in the current state.

**The streetlight effect** comes from the old joke about the drunk looking for his keys under the streetlamp not because he dropped them there but because the light is better. We debug where the tools are convenient, the logs are verbose, the code is familiar — not where the bug actually is. The bug is in the third-party library with no logging, the async callback with no stack trace, the production-only code path, the config layer, the network — all the dark places where investigation is annoying. The counter-discipline is to *go where the symptom is, even when the light is bad.* If the corruption first appears at the boundary with an external service, instrument that boundary even though it is painful, rather than re-examining your nicely-logged business logic for the fifth time because it is comfortable. Ask honestly: "am I looking here because the evidence points here, or because looking here is easy?"

**Sunk cost** is the trap that turns a thirty-minute mistake into a four-hour one. You have invested two hours in a theory. It is starting to look wrong — the experiments are not landing — but you have *two hours in it*, so you keep going, refining the dead theory, adding epicycles, because abandoning it means admitting the two hours were wasted. They were. They are gone either way; that is what "sunk" means. The counter-discipline is to *drop a theory the moment it is falsified, with zero loyalty to it,* and the lab notebook (next section) is what makes this possible — when it is written down that "H1: killed at 09:25 by the assert that passed," you cannot fool yourself into wandering back to it, and you cannot pretend it is still alive because you are emotionally attached to your two hours.

There is a sixth trap that is so important it gets its own section: confusing correlation with causation.

## 7. Correlation is not causation: the fix that was a coincidence

You change something. The bug goes away. You ship. This is the most seductive false-positive in all of debugging, and it has ended more incidents *temporarily* than any other move — temporarily, because the bug was never actually fixed; it just stopped reproducing for reasons unrelated to your change, and it will be back, usually at a worse time.

The mechanism behind this trap is worth making concrete, because understanding *why* the coincidence happens is what immunizes you. Many of the nastiest bugs are intermittent — races, timing-dependent failures, load-dependent corruption, heisenbugs whose probability of manifesting depends on the exact interleaving of threads or the exact timing of I/O. When a bug only fails 5% of the time, *any* change you make has a 95% chance of "fixing" it on the next single run, purely by luck, even if the change is completely unrelated to the cause. You add a log line — which slightly changes timing — and the race stops reproducing, not because the log fixed the race but because the extra microseconds shifted the interleaving probability. You add a `sleep`, a retry, a reorder, a defensive null check on an unrelated field, and the symptom vanishes. You did not fix anything. You perturbed the timing, and the dice rolled your way on the one run you watched.

This is why "it works now" after a single run is nearly worthless evidence for an intermittent bug. The probability of catching a 5%-failure bug in $n$ runs is $1 - 0.95^n$, so one run only has a 5% chance of even showing the bug — meaning a 95% chance of a false "fixed." To get to 99% confidence you need to *not* see the failure across enough runs that the failure would almost surely have appeared if it were still live: solving $0.95^n < 0.01$ gives roughly $n > 90$ runs. One green run proves nothing; ninety green runs is evidence.

![A two-column before-and-after contrast showing a retry believed to have fixed a bug because the symptom lined up, versus the same suspect proven causal by toggling a flag off and on across ten runs each](/imgs/blogs/hypothesize-and-falsify-not-stare-and-hope-8.png)

The before-and-after above is the discipline that breaks the trap: *toggling.* A correlation becomes a proven cause only when you can turn the suspected cause off and watch the bug return, then turn it on and watch the bug leave — and do it enough times that luck is excluded. On the left, a retry that "seemed to fix it" and came back Wednesday at peak load. On the right, the same suspect proven with the flag off (bug present 10 of 10 runs) and on (bug absent 10 of 10 runs). That is causation: the symptom tracks the toggle deterministically across repeated trials. If you cannot make the bug reappear by undoing your change, you have not proven your change fixed it — you have a correlation, and correlations on intermittent bugs are usually coincidences. The general rule, which I will die on: **you do not understand a bug until you can turn it on and off at will.** If you can reproduce it deterministically and suppress it deterministically with the same lever, you own it. If "it just stopped happening," you own nothing and it will be back.

#### Worked example: the retry that fixed nothing

A payment-reconciliation job occasionally double-charged a customer — roughly 1 in 200 runs. An engineer noticed the failures clustered around a downstream timeout, added a retry with backoff around the charge call, ran the job a few times, saw no double-charge, and shipped. Eleven days later, at month-end peak volume, the double-charge rate spiked to 1 in 30 and finance escalated.

What went wrong, in falsification terms: the original "fix" was never falsified. The retry was added because the failures *correlated* with timeouts, and after shipping it the engineer observed three clean runs and concluded "fixed." But three clean runs of a 1-in-200 bug is the expected outcome 98.5% of the time even if nothing changed — the probability of seeing the failure in three runs is $1 - 0.995^3 \approx 1.5\%$. The clean runs were not evidence of a fix; they were evidence of nothing. Worse, the retry was *causally backwards*: the real bug was that the charge succeeded downstream but the success response was lost to the timeout, so the retry *re-charged*. The retry did not fix the double-charge; under load it *caused more of them*, which is exactly why the rate went up at peak, not down.

The disciplined version: form the hypothesis "the charge succeeds but the ack is lost, and any retry double-charges." Predict: "if true, then toggling the retry OFF should make the double-charge rate go to zero, because without a retry there is no second charge." Toggle it off, run 500 times: zero double-charges (and a different symptom appears — single failed charges on timeout, which is the *correct* behavior to then fix with idempotency, not blind retry). Toggle on, run 500: double-charges return at the load-dependent rate. The toggle proves the retry is causal for the double-charge. The actual fix is an idempotency key on the charge so a retry is safe — see [idempotency and deduplication: making at-least-once safe](/blog/software-development/message-queue/idempotency-and-deduplication-making-at-least-once-safe) for why this is the correct pattern, not blind retry. The number that matters: the bug went from "fixed" (1-in-200, lurking) to genuinely 0-in-2000 once the idempotency key was added and verified by toggling, where before it had been *worsened* by a change everyone believed had fixed it.

## 8. The debugging log: a lab notebook for a six-hour bug

Everything above assumes you can keep track of which hypotheses are alive, which are dead, and why. On a five-minute bug you can hold that in your head. On a six-hour bug — or a bug you put down on Friday and pick up Monday, or a bug three people are working — you cannot, and the failure mode is vicious: you re-test theories you already killed, you forget the result of an experiment you ran two hours ago, you lose the thread entirely and start over. The cure is embarrassingly low-tech: write it down. Keep a debugging log, a lab notebook, where each entry is a hypothesis, the experiment, and the result.

![A horizontal timeline of a debugging session logging hypotheses and results from stale-cache through bad-input and wrong-environment to a replica-lag root cause confirmed by toggling](/imgs/blogs/hypothesize-and-falsify-not-stare-and-hope-6.png)

The timeline above is a real-shaped notebook for a single session: 09:10 H1 stale cache, bypassed, still bad — killed. 09:25 H2 bad input, assert passes — killed. 09:40 H3 wrong environment, print the host, it is the wrong replica — alive. 09:55 test against the prod primary, bug reproduces. 10:20 replica lag is the root cause. 10:40 toggle confirms the fix. Notice what the log buys you: at 09:40 you do not go back and re-bypass the cache, because the log says cache is dead. You do not re-check the input, because the log says input is clean. The written record enforces the sunk-cost counter-discipline automatically — a killed hypothesis stays killed because it is on the page with a strikethrough and a reason. It also makes the bug *resumable*: anyone, including future-you, can pick up the notebook and see exactly which half of the space is left.

The format does not need to be fancy. A plain text file, a code comment block, a sticky note for a short one. The non-negotiable fields are: the hypothesis (falsifiable, with a prediction), the experiment you ran, the observed result, and the verdict (killed / survived / narrowed-to). Here is a template I actually use, kept in a scratch file next to the code — written as a Python module docstring so it lives in the repo and stays version-controlled with the fix:

```python
NOTEBOOK = """
BUG: invoice total +1 cent, intermittent, always high
REPRO: GET /invoice/8841 ; fails ~1 in 5 ; faster under concurrent load
SPACE: code(sum,round) | data(items list) | config(currency,rounding) | env(db replica)

H1  cache returns stale subtotal
    predict: bypass cache -> bug gone
    exp:     ?cache_bypass=1 on /invoice/8841 x20
    result:  still wrong 4/20         -> KILLED, cache innocent
H2  one line item double-counted
    predict: len(summed) == displayed+1
    exp:     log n_items vs displayed
    result:  equal, no dup            -> KILLED, not double-counting
H3  rounding mode half-up not banker
    predict: raw subtotal already +1c BEFORE round
    exp:     print subtotal pre-round
    result:  subtotal clean, error appears AFTER round -> SURVIVES, narrow to round()
H4  round() called per-line then summed (fence-post)
    predict: error scales with count of half-cent lines
    exp:     craft invoice, 1 half-cent line vs 5
    result:  +1c with 5 half-cent lines, +0 with 0 -> CONFIRMED, root cause
FIX: round once on the sum, not per line. toggle old/new: old +1c 9/20, new 0/2000.
"""
```

That is the whole discipline on one page. Look at how each entry has a prediction *before* the result, which is what stops you from rationalizing whatever you see into support for whatever you wanted (confirmation bias has no room to operate when the prediction is written before the observation). Look at how H1 and H2 are struck dead with reasons, so you never wander back. Look at the SPACE line at the top — that is your search space written out so you can watch it shrink: each killed hypothesis crosses off a region, and you can literally see when you are down to one. And look at the FIX line: it is not "round once on the sum" full stop — it is that plus the toggle result (old +1c in 9 of 20, new 0 in 2000), because a fix is not proven until you have turned the bug back on and off. For a deeper take on instrumenting systems so this kind of evidence is available at all, the [observability by design](/blog/software-development/system-design/observability-metrics-logs-traces-by-design) post in the system-design series covers the logging and tracing infrastructure that makes a production lab notebook possible.

## 9. A full worked example: "it can't be the database"

Let me run the assumption-audit discipline end to end on the canonical "it can't be that" bug, because it shows every piece working together.

#### Worked example: the assumption that was wrong

The symptom: a reporting endpoint returns yesterday's numbers instead of today's, but only in the staging environment, and only intermittently — maybe 1 request in 3. Production is fine. Local is fine. The engineer's immediate theory, stated with total confidence: "it's a caching bug, the report is being served from a stale cache." Reasonable! Stale data, intermittent, environment-specific — caching fits.

So they spend ninety minutes on the cache. They bypass it (bug persists — should have been the falsification, but they assume they bypassed it wrong and try again). They flush it. They add cache-busting headers. They read the cache library's source. The bug remains, stubbornly, 1 in 3. The whole time, there is a sentence they keep saying: "it can't be the database, both replicas have the same data, I checked." That sentence is the flashing arrow.

The disciplined reset, after recognizing the stall: run an assumption audit. Write the list.

1. The cache is involved. (Cost to verify: bypass and check. *Already done — bypassing did not fix it. This belief is FALSIFIED and I ignored it.*)
2. The input request is identical between good and bad responses. (Cheap: log the full request on both. Result: identical. Killed as a difference.)
3. The query is correct. (Cheap: log the exact SQL executed. Result: correct, identical on good and bad. Killed.)
4. The database returns the right rows. (Cheap: log the rows returned by the query, with a timestamp column. *Result: on the bad responses, the rows are real but stale — yesterday's. On good responses, today's. The database itself is handing back stale data.*)
5. "Both replicas have the same data." (The belief I was sure of. Cost to verify: actually query each replica directly and compare. *Result: they do NOT. One replica is lagging the primary by hours.*)

There it is. The symptom was intermittent and environment-specific not because of a cache but because staging's load balancer round-robins reads across two database replicas, and one of them had fallen hours behind on replication. Roughly 1 request in 3 hit the lagging replica and got stale rows; production used a single replica so it never showed; local used one database so it never showed. "It can't be the database" was wrong in the most literal way — it was *entirely* the database — and the engineer had "checked" it by glancing at a monitoring dashboard that aggregated the replicas and hid the lag.

The mechanism, made concrete, is the *why* this bug class even exists: in a primary-replica setup with asynchronous replication, a write to the primary is acknowledged before it has propagated to the replicas. There is a window — replication lag — during which the replicas serve data older than the primary. If your reads are load-balanced across replicas with no read-your-writes guarantee, a read can land on a replica that has not yet caught up, and you get the past. The lag is normally milliseconds and invisible; under replication strain (a big backfill, a slow disk, a network hiccup) it stretches to hours and becomes your bug. This is a real and common failure mode; the [database replication: sync, async, logical, physical](/blog/software-development/database/database-replication-sync-async-logical-physical) material in the database series goes deep on the mechanism and the read-your-writes patterns that prevent it.

The falsification that should have happened at minute two: bypassing the cache did not fix the bug. That single experiment *killed the cache hypothesis* and the engineer rationalized it away ("I must have bypassed it wrong") because of confirmation bias and sunk cost — they were committed to the cache theory and would not let a clean negative result kill it. The whole ninety minutes was the cost of not respecting a falsification. The assumption audit recovered it by forcing the belief "it can't be the database" onto the list as a *testable item* — query each replica directly — which is exactly the thing the engineer had been treating as too obvious to test. The bug lived in what they were sure of.

The numbers, to make the proof honest: before the fix, the staging endpoint returned stale data on 31% of requests (measured by logging the max timestamp in the returned rows across 400 requests: 124 of them were over an hour stale). After routing reads for this endpoint to the primary (or, better, adding a read-your-writes session guarantee), stale responses dropped to 0 of 2,000 requests. Toggling confirmed causation: force reads back to the lagging replica and the 31% returns; force them to the primary and it is 0%. The bug was turned off and on at will — which is the standard for "understood and fixed."

## 10. The mechanism: why your perception of the code is unreliable

It is worth stepping back to the deepest *why* under this whole post, because it justifies the entire discipline. Why can't you just read the code and see the bug? Why is staring so useless? The mechanism is that the code you read is not the code that runs, in several stacked senses, and your mental model of execution is a lossy approximation that omits exactly the things bugs live in.

When you read code, you simulate it in your head — but your head runs a simplified, optimistic interpreter. It executes the happy path. It assumes inputs are well-formed. It assumes the function is called with the arguments the name suggests. It runs single-threaded, so it cannot see the interleaving that produces a data race (your mental interpreter has no concept of two threads touching the same memory between two of your "steps"). It assumes the config is the default. It does not model the compiler's reordering, the CPU's out-of-order execution, the allocator reusing a freed block, the network dropping a packet, the clock jumping at a leap second, the float that cannot represent 0.1. Every one of those is a real mechanism that produces real bugs, and your reading-the-code mental interpreter is blind to all of them. That is *structurally* why staring fails: you are running a model of the program that has been pre-edited to exclude the conditions under which it breaks.

This is also why the experiment beats the simulation every time. An experiment runs the *real* interpreter — the actual CPU, the actual allocator, the actual replicas, the actual interleaving — and reports what really happened, with none of your optimistic omissions. When you print the resolved config and it is the wrong host, the real machine just told you something your mental model would have asserted was impossible. When you toggle a flag and the bug tracks it, the real machine just proved causation that no amount of reading could establish. The discipline of this whole post — hypothesize, predict, experiment, observe — is fundamentally a discipline of *not trusting your mental interpreter and asking the real one instead.* You hypothesize with your head (cheap, fast, but unreliable) and you adjudicate with an experiment (the ground truth). The bug is, by definition, in the gap between your model and the machine. The only way to find that gap is to make a prediction with your model and check it against the machine. Where they disagree, that is your bug.

This connects back to the assumption audit with full force. Each item in the audit is a place where your mental interpreter quietly substituted an assumption for a fact — "input valid," "function called," "config default," "replicas consistent." The audit is the systematic act of replacing each assumed fact with a measured one, closing the gaps in your model one cheap experiment at a time, until the model and the machine agree everywhere except the one place that is the bug.

## 11. War story: famous bugs that were "impossible"

The history of catastrophic software bugs is, to a remarkable degree, a history of people being certain about something that was false — the "it can't be that" trap at industrial scale. Three real cases, accurately:

**The Therac-25 race condition (1985-1987).** The Therac-25 was a radiation therapy machine that, in at least six incidents, delivered massive radiation overdoses, killing or seriously injuring patients. A central cause was a race condition: if an experienced operator typed the treatment parameters very fast, a particular sequence of edits could complete before a setup routine finished, leaving the machine in an inconsistent state where the high-power beam fired without the beam-spreader in place. The bug was timing-dependent and only reproduced with fast operators, which is exactly why it survived testing — the developers' mental interpreter ran the input slowly and single-threaded, the way they typed it, and never saw the interleaving. The hypothesis "operator speed affects the outcome" would have been falsifiable and, if entertained, fatal to the bug; but "it can't be the timing, the software checks the state" held, because the check itself had the race. The lesson in our terms: a symptom that correlates with *speed of input* or *concurrency* is screaming "race," and a race is invisible to the single-threaded simulation in your head. You must reproduce it under the real interleaving. (The general mechanism — why a race produces a torn or inconsistent state because there is no happens-before edge ordering the two operations — is a deep enough topic that this series gives it its own posts on data races and heisenbugs.)

**The Ariane 5 Flight 501 explosion (1996).** Forty seconds after launch, the rocket self-destructed. Root cause: a 64-bit floating-point value (horizontal velocity) was converted to a 16-bit signed integer; the value was larger than 32,767 and the conversion overflowed, raising an unhandled exception that shut down the navigation computer, then its backup (running identical code, failing identically), and the loss of guidance triggered self-destruction. The code that overflowed was *correct for Ariane 4* — it had been reused on the assumption that the flight profile was similar. "This code works, it flew on Ariane 4" was the unchecked assumption; Ariane 5's higher horizontal velocity falsified it instantly. The assumption-audit item that would have caught it: "the input range on Ariane 5 is within the range this conversion was validated for" — cheap to check, never checked, because the code was *trusted*. Bugs live in trusted code.

**The Knight Capital trading loss (2012).** In 45 minutes, a deployment bug caused Knight Capital to lose about \$440 million and effectively destroyed the company. A new feature reused an old, repurposed feature flag; the deploy went to only seven of eight servers; the eighth ran old code that interpreted the flag the *old* way, firing a defunct test routine that placed millions of unintended live orders. The "it can't be that" here was "the deploy succeeded" — the assumption that all eight servers were running the new code. One server was not. An assumption audit item — "the running code on every server is the code I deployed" — was the highest-yield check in the world that day and was not made. This is the single most important assumption in production debugging, stated earlier: *is the code that's running the code I think is running?* Knight Capital is what it costs to assume yes.

The thread through all three: each was a confidently-held belief — "the software checks the state," "this code already flew," "the deploy succeeded" — that was false, untested, and load-bearing. The discipline that defeats all three is identical: write the belief down as a falsifiable hypothesis and test it, *especially* the one you are most sure of. The bug is in what you are sure of, at every scale from a one-cent invoice to a four-hundred-million-dollar morning.

## 12. How to reach for this (and when not to)

The hypothesis-driven discipline is the core of debugging, but like every tool it has a cost and a wrong context. Let me be decisive about both.

**Reach for full hypothesize-predict-experiment when:** the bug is non-trivial (more than a misspelled variable a glance catches); the search space is large (many possible causes, multiple subsystems, intermittent); you have already stared for more than ten minutes without progress (the stare-timer is a real and useful trigger — when you notice you have been re-reading, *stop and write a hypothesis*); the bug is intermittent or load- or timing-dependent (where intuition is worst and toggling-for-causation is essential); or anyone else needs to be able to pick up the investigation (the log makes it resumable and shareable).

**Do not over-engineer when:** the bug is obvious and local — if the stack trace points at line 42 and line 42 has an off-by-one you can see, fix it, do not write a lab notebook. The discipline is for *uncertainty*; when there is little uncertainty, skip the ceremony. A one-character typo does not need a falsification loop. The skill is calibrating effort to uncertainty: cheap bugs get a glance, expensive bugs get the full method. Spending twenty minutes formalizing hypotheses for a bug you could fix in two is its own kind of waste.

**A few sharp don'ts I have learned the hard way:**

- **Don't run an experiment that doesn't discriminate.** Before you run anything, ask "what will I conclude if it comes out yes? if no?" If both answers are "I still don't know," the experiment is worthless — design a better one. The most common waste is adding a print that every one of your live hypotheses predicts the same value for.
- **Don't trust a single run of an intermittent bug, in either direction.** One green run does not mean fixed; one red run does not mean your last change broke it. Repeat until the probability of a false conclusion is small (the $1 - 0.95^n$ math). For a 5%-failure bug, ninety clean runs is the threshold for 99% confidence, not three.
- **Don't keep a theory alive past its falsification out of sunk cost.** When the experiment kills it, kill it. The two hours are gone regardless; the only question is whether you waste a third.
- **Don't skip the assumption audit on a stalled bug.** The moment you have been stuck for thirty minutes, the highest-yield move is almost never "stare more" — it is "list what I'm assuming and test the cheap ones," because by minute thirty the bug is provably in something you have not yet questioned.
- **Don't confuse "the symptom stopped" with "I fixed the cause."** Prove causation by toggling. If you cannot make the bug come back by undoing your change, you have not earned the word "fixed."
- **Don't formalize when intuition is faster and the cost of being wrong is low.** A senior debugger uses intuition to *generate* hypotheses fast (that is what experience buys) and then uses the discipline to *adjudicate* them honestly. Intuition proposes; experiment disposes. The discipline is not anti-intuition; it is the check that keeps intuition honest.

The relationship to the rest of the loop is worth restating. Hypothesizing without [reproducing](/blog/software-development/debugging/stop-guessing-the-scientific-method-of-debugging) first is hard, because you cannot run a clean experiment on a bug you cannot trigger on demand — get a reliable reproducer before you start the falsification loop, or your experiments will themselves be intermittent and uninterpretable. And once your hypotheses have narrowed the space to "a regression between two known-good and known-bad states," the discipline becomes mechanical and you should hand it to a tool: [binary-search your bug with bisection](/blog/software-development/debugging/binary-search-your-bug-with-bisection) automates the halving across a commit range with `git bisect run`. Hypothesis-driven debugging is the general skill; bisection is its most automated, highest-leverage special case.

## War story addendum: the heisenbug that the print fixed

One more short case because it crystallizes the correlation trap. A multithreaded queue consumer crashed about once an hour in production with a use-after-free. An engineer added a `printf` near the suspected line to log the pointer — and the crashes stopped. For two days, the build with the `printf` ran clean. Everyone relaxed.

This is the classic heisenbug, and the falsification analysis is precise. The crash was a data race with a tiny timing window. The `printf` did nothing to fix the race; it added a few microseconds of latency and a lock (stdio is internally synchronized) that shifted the thread interleaving so the window almost never lined up. The bug was not fixed; its *probability* was lowered by a coincidental timing change. The correlation — "added print, crash stopped" — was real and causal in the narrow sense that the print did change behavior, but it did not address the cause, and the moment load patterns shifted, the window realigned and the crashes returned.

The disciplined move would have been: do not trust "crashes stopped" from a print, because a print is a notorious timing-perturbing change for exactly the bug class (races) that produces intermittent crashes. Instead, falsify causation directly with a tool that detects the race regardless of timing — run under ThreadSanitizer (`-fsanitize=thread`, or `go test -race` for Go), which instruments memory accesses and reports the racing pair with both stack traces even on a run where the crash does *not* manifest. TSan does not rely on the race actually triggering; it detects the *absence of a happens-before edge* between the two accesses, which is the mechanism of the bug, present on every run whether or not the timing aligns. That turns an intermittent, timing-dependent, print-maskable crash into a deterministic, always-reported diagnostic. The print "fixed" it by correlation; TSan *found* it by causation. The fix — a lock or an atomic establishing the missing happens-before edge — was then verified the right way: TSan clean across 2,000 runs, and the crash rate from once-an-hour to zero over a week, with the race no longer reported on any run.

## Key takeaways

- **Replace staring with a loop.** Hypothesize a falsifiable claim, predict an observation, run the cheapest experiment that produces or destroys it, observe, update. Staring produces no information; an experiment partitions the search space. If an action does not narrow the space, it is not debugging.
- **A real hypothesis predicts an observation.** "Something's wrong with the cache" is a feeling. "If I bypass the cache the bug disappears" is a hypothesis — it can be confirmed or killed by one experiment. If you cannot write "if true, then I will observe X," sharpen until you can.
- **Try to falsify your favorite theory, not confirm it.** Confirmation is weak and endless; one falsification is final. Design the experiment that would prove you wrong, and run that one first. A theory that survives a genuine attempt to kill it is worth a hundred that merely seem consistent.
- **Design for information gain.** Prefer the cheapest experiment that discriminates between your live hypotheses and splits the space most evenly. The midpoint probe, the subsystem bypass, the flag toggle, and the targeted assert beat random prints and single-stepping for searching a large space.
- **Run the assumption audit on any stalled bug.** List everything you believe is true — input valid, code ran, config loaded, deploy succeeded, replicas consistent — and test the cheap ones first. The bug lives in the belief you never wrote down. "I already checked that" means "I checked it under conditions that no longer hold."
- **"It can't be that" is a flashing arrow.** The region you refuse to suspect is where the bug hides. Re-verify the thing you are most sure of, especially the trusted code, the obvious config, and whether the running code is the code you think is running.
- **Correlation is not causation; prove the fix by toggling.** A change that makes the symptom vanish may be a coincidence, especially for intermittent bugs where any perturbation "fixes" it by luck. You have not fixed a bug until you can turn it off and on at will with the same lever.
- **One run proves nothing for an intermittent bug.** The probability of catching a 5%-failure bug in $n$ runs is $1 - 0.95^n$; one green run is 95% likely even if nothing changed. Repeat until the false-conclusion probability is small — roughly ninety clean runs for 99% confidence on a 1-in-20 bug.
- **Keep a lab notebook.** Hypothesis, prediction, experiment, result, verdict — written before you observe, so confirmation bias has no room and so a six-hour or multi-person bug stays resumable and you never re-test a dead theory.
- **Calibrate effort to uncertainty.** Trivial local bugs get a glance; large, intermittent, multi-subsystem bugs get the full method. Intuition proposes hypotheses fast; the discipline adjudicates them honestly. The method is not anti-intuition — it is the check that keeps intuition from lying to you.

## Further reading

- *Debugging: The 9 Indispensable Rules for Finding Even the Most Elusive Software and Hardware Problems* by David J. Agans — the canonical short book; "quit thinking and look" and "make it fail" are this post in two slogans.
- *Why Programs Fail: A Guide to Systematic Debugging* by Andreas Zeller — the academic treatment of scientific, hypothesis-driven debugging and delta debugging; the rigorous version of everything here.
- Karl Popper, *The Logic of Scientific Discovery* — the source of falsification; you do not confirm theories, you fail to refute them, and that asymmetry is the whole engine.
- Nancy Leveson and Clark Turner, "An Investigation of the Therac-25 Accidents" (IEEE Computer, 1993) — the definitive analysis of the race condition and the "it can't be the software" certainty that let it kill.
- The ThreadSanitizer documentation (the LLVM/Clang `-fsanitize=thread` and Go `-race` manuals) — how to detect a race by its missing happens-before edge regardless of whether the timing aligns on a given run.
- [Stop guessing: the scientific method of debugging](/blog/software-development/debugging/stop-guessing-the-scientific-method-of-debugging) — the series intro that frames the full observe, reproduce, hypothesize, bisect, fix, prevent loop this post sits inside.
- [Binary-search your bug with bisection](/blog/software-development/debugging/binary-search-your-bug-with-bisection) — the automated, highest-leverage special case of the halving experiment, applied across a commit range with `git bisect run`.
- [Observability by design: metrics, logs, and traces](/blog/software-development/system-design/observability-metrics-logs-traces-by-design) — the instrumentation that makes a production lab notebook and a cheap discriminating experiment possible in the first place.
