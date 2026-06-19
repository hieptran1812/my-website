---
title: "Off-by-One and Boundary Bugs: Why the Edges Are Where Everything Breaks"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Learn why off-by-one errors and boundary bugs are inevitable, how the fence-post and half-open-interval problems create them, and how to debug with edge inputs and prevent them forever with property-based testing."
tags:
  [
    "debugging",
    "software-engineering",
    "off-by-one",
    "boundary-conditions",
    "property-based-testing",
    "binary-search",
    "pagination",
    "edge-cases",
  ]
category: "software-development"
subcategory: "Debugging"
author: "Hiep Tran"
featured: true
readTime: 53
image: "/imgs/blogs/off-by-one-and-boundary-bugs-1.png"
---

There is an old joke that has been told so many times nobody remembers who said it first: there are two hard things in computer science, cache invalidation, naming things, and off-by-one errors. The joke works precisely *because* it is off by one. You expect two items in the list and you get three, and the third one is the punchline. That tiny structural surprise — one more, or one fewer, than you counted on — is the exact shape of the bug class this post is about, and it is responsible for an astonishing amount of grief in real systems.

I have lost more hours than I want to admit to bugs that lived at an edge. A report that was correct for every customer except the one who happened to have exactly fifty orders, because page two re-fetched row fifty. A binary search — *binary search*, the canonical "simple" algorithm — that ran fine on a list of a thousand items and hung forever on a list of one. A billing job that double-charged on the last day of the month because the time window used `<=` on the end instead of `<`. A buffer copy that worked in every test and then, on an input that was exactly the buffer's capacity, wrote one byte past the end and corrupted the heap so quietly that the crash came ten thousand lines later in a totally unrelated function. None of these were exotic. Every one of them lived at a boundary: empty, single, full, the last element, the wrap point, midnight, the page seam. The middle of the range was always fine. The edge was where everything broke.

This is not bad luck. It is structural, and once you see the structure you can hunt these bugs deliberately instead of stumbling into them. The reason boundaries are where bugs live is that the boundary is where two different ways of counting collide — where "the number of things" meets "the number of gaps between things," where a half-open interval `[start, end)` meets a closed one `[start, end]`, where `len` meets last-index, where the loop's first iteration and its last iteration do something different from every iteration in between. The figure below is the seed of all of it: the fence-post problem, where a hundred feet of fence with a post every ten feet needs eleven posts but only ten rails, and the human brain — and a surprising amount of code — wants to say ten.

![A fence-post diagram showing that a span needs eleven posts but only ten rails, illustrating that element counts and gap counts always differ by one](/imgs/blogs/off-by-one-and-boundary-bugs-1.png)

By the end of this post you will be able to do three things. First, *recognize* the boundary in any piece of code on sight — the loop guard, the interval convention, the index arithmetic — so you can predict where it will break before it does. Second, *debug* a boundary bug methodically: reproduce it with the edge input first (empty, single, max), shrink the failing case to the smallest input that still triggers it, trace the index at the boundary iteration, and binary-search which input *size* flips the behavior. Third, *prevent* the entire class with property-based testing that hammers the edges, explicit boundary unit tests, invariant assertions, and the single most effective habit of all — picking one interval convention (half-open) and holding it everywhere. This is a direct application of the series' spine: observe the symptom, reproduce it deterministically with the right input, form a falsifiable hypothesis about *which* edge, bisect the gap between what you believe and what is true, fix, and then make the bug impossible to reintroduce. If you have not read the intro to that method, start with [the scientific method of debugging](/blog/software-development/debugging/stop-guessing-the-scientific-method-of-debugging); this post is the boundary-specialist chapter.

## 1. The fence-post problem: counting things versus counting gaps

Start with the oldest version of this bug, the one that predates computers. You are building a straight fence one hundred feet long, and you will put a post every ten feet. How many posts do you need? The arithmetic that leaps to mind is one hundred divided by ten, which is ten. And that answer is wrong by one: you need *eleven* posts. One at zero feet, one at ten, twenty, thirty, all the way to one hundred. The division gave you the number of *segments* — the rails between posts — and there are ten of those. But you asked for posts, and there is always one more post than there are gaps between posts.

This is the fence-post error, and it is the conceptual root of nearly every off-by-one bug you will ever write. The trap is that two quantities that *feel* like the same number are actually different by exactly one: the count of items, and the count of intervals between items. A week has seven days but the span from Monday to Sunday is six days wide. A ruler twelve inches long has thirteen tick marks if you count both ends. A list with five elements has four "between" positions where you might insert something, but six positions total if you count "before the first" and "after the last." Every time your code conflates *how many things* with *how many gaps*, you are one step from a fence-post bug.

In code this shows up the instant you write a loop or compute a length. Consider the most innocent line imaginable:

```python
# Print the numbers from low to high, inclusive.
def print_range(low, high):
    for i in range(low, high):
        print(i)
```

Call `print_range(1, 10)` expecting the numbers one through ten, and you get one through *nine*. Python's `range(low, high)` is half-open — it stops *before* `high` — so it produces `high - low` numbers, which is nine here, not ten. The function's docstring promised "inclusive," but the implementation counted gaps. To get ten numbers you write `range(low, high + 1)`, and now the `+ 1` is the fence post you forgot. The bug is not in the arithmetic; the arithmetic is correct for *counting gaps*. The bug is the mismatch between what you said you wanted (a count of items, inclusive) and what the tool gives you (a count of gaps, exclusive).

The reason this is worth dwelling on is that it tells you where to *look* when a boundary bug appears. Whenever a result is too big or too small by exactly one, the first hypothesis should be: *somewhere, a count of items got used where a count of gaps was needed, or vice versa.* That single framing collapses a huge fraction of off-by-one mysteries into a quick check. Did you write `n` where you meant `n - 1`? Did you size a buffer by the number of elements but index it by the number of separators? Did a date range use the number of days but iterate over the number of midnights between them?

There is a deeper systems reason these never go away, and it is worth stating plainly because it kills the temptation to think "I'll just be more careful." The reason is that *most programming interfaces, deliberately, count gaps.* Array indices start at zero, so the valid indices of an `n`-element array are `0` through `n - 1` — the last index is one less than the length. Pointer arithmetic in C measures distances, which are gaps. Iterator pairs in C++ mark a beginning and a one-past-the-end, which is a gap-based bound. This is not an accident or a mistake; as we will see, half-open gap-based bounds are mathematically the *better* choice, which is exactly why the whole ecosystem adopted them. But it means your code is constantly translating between human "how many" (items) and machine "where" (gaps), and every translation is a chance to be off by one.

## 2. Half-open, closed, and one-indexed: the convention war

If the fence-post problem is the *why*, interval conventions are the *how it actually bites you in production*. An interval is just a range of values with a low end and a high end, and there are three common ways to write down which endpoints are included. Pin these down now, because mixing two of them is the single most common source of off-by-one bugs in real codebases.

A *half-open* interval, written `[start, end)`, includes `start` but excludes `end`. The square bracket means "included," the parenthesis means "excluded." So `[0, n)` is the set `{0, 1, 2, ..., n-1}` — exactly `n` values, and notice the largest one is `n - 1`. A *closed* interval `[0, n]` includes *both* ends: `{0, 1, ..., n}` — that is `n + 1` values. A *one-indexed* closed interval `[1, n]` is what humans use when they count out loud: `{1, 2, ..., n}` — `n` values, no zero. The figure below stacks these three conventions and the languages that use them, because the entire bug class is "the code at this layer assumed one convention and the code at that layer assumed another."

![A layered diagram comparing half-open, closed, and one-indexed interval conventions and the languages that use each, showing that mixing them drops or doubles one boundary element](/imgs/blogs/off-by-one-and-boundary-bugs-2.png)

Here is the load-bearing fact: **the overwhelming majority of modern languages and APIs use half-open `[start, end)`.** Python slices: `xs[0:n]` gives you the first `n` elements and stops before index `n`. Python's `range(start, stop)`. C++ iterators: `begin()` points at the first element and `end()` points at one past the last, so a loop runs while the iterator is `!= end()`. Rust ranges `0..n`. Go and Java `for (i = 0; i < n; i++)` with the strictly-less-than guard. JavaScript `Array.prototype.slice(start, end)`. Even `substring(start, end)` in most languages is half-open. The convention is nearly universal, and it is universal for good reasons we will get to.

So why do half-open intervals win? Three properties, and each one directly prevents a boundary bug. First, **the length is trivially `end - start`** — no `+ 1`, no fence post to forget. `[3, 8)` has `8 - 3 = 5` elements, done. With a closed interval you would have to remember `end - start + 1`, and the `+ 1` is the bug. Second, **adjacent ranges tile perfectly with no overlap and no gap.** The ranges `[0, 5)` and `[5, 10)` cover `0` through `9` with no element shared and none missing, because the `5` that one range excludes is exactly the `5` the next range includes. This is *the* property that makes pagination, chunking, sharding, and range partitioning correct — and as we will see, it is exactly what a closed convention destroys. Third, **an empty range is just `[k, k)`** — start equals end, zero elements, no awkward "how do I represent nothing" special case. A closed interval has no clean way to say "empty"; `[5, 4]` is nonsense.

Now watch what happens when you mix conventions. Suppose a database query returns rows where the `id` is `> last_seen AND id <= last_seen + page_size`. The first clause is exclusive (half-open on the low end), the second is *inclusive* (closed on the high end). Page one fetches ids `1` through `50` with `last_seen = 0`. Page two sets `last_seen = 50` and fetches `id > 50 AND id <= 100`, so `51` through `100`. So far so good — but only because the low end was correctly exclusive. Now flip it: a developer "fixes" the boundary by making the *low* end inclusive too, `id >= last_seen`, reasoning that "we want to include where we left off." Page one ends at id `50`. Page two now fetches `id >= 50`, which re-reads row `50`. Every page seam now duplicates a row. The mismatch is the whole bug: one end was half-open, the other was closed, and the boundary element fell into both ranges or neither. The figure below shows the seam, and the fix is to keep the boundary half-open so the pages tile.

![A before-and-after comparison of a paginated query where a closed guard re-reads the boundary row at every page seam versus a half-open guard whose pages tile into the full set](/imgs/blogs/off-by-one-and-boundary-bugs-4.png)

The same off-by-one shows up at the database layer in countless paginated APIs, and it is worth noting that keyset pagination (filtering by `id > last_seen` rather than `OFFSET`) is the half-open-friendly approach precisely because it avoids the seam arithmetic that `OFFSET`/`LIMIT` invites; the database-side trade-offs are covered in the [database series on indexing and queries](/blog/software-development/database/b-trees-how-database-indexes-work).

The practical rule that falls out of this is short enough to tattoo on the back of your hand: **pick half-open, and never let a closed convention leak in.** When you read code and see a `<=` in a loop guard or a `+ 1` in an index computation, your suspicion meter should jump, because those are the syntactic fingerprints of a closed convention smuggled into a half-open world. They are not *always* wrong — sometimes you genuinely want the last element — but they are where the bugs cluster, and they earn a second look every time.

| Convention | Notation | Element count | Last index/value | Empty case | Used by |
| --- | --- | --- | --- | --- | --- |
| Half-open | `[start, end)` | `end - start` | `end - 1` | `[k, k)` | Python slices, C++ iterators, `range`, Go/Java `<` loops |
| Closed | `[start, end]` | `end - start + 1` | `end` | none (no clean form) | math notation, some date ranges, `BETWEEN` in SQL |
| One-indexed closed | `[1, n]` | `n` | `n` | `[1, 0]` (awkward) | human counting, Lua, MATLAB, line numbers |

## 3. The loop guard: `<` versus `<=`, and the first and last iteration

Most off-by-one bugs are born in a single character: the comparison operator in a loop's continuation condition. Get it right and the loop runs exactly the right number of times. Get it wrong by one symbol and you either skip the last element or run one iteration too many — and "one iteration too many" on an array means reading or writing out of bounds, which is the gateway to crashes and security holes.

Take the canonical C-style loop:

```c
int sum = 0;
for (int i = 0; i < n; i++) {
    sum += arr[i];   /* visits arr[0] .. arr[n-1], which is correct */
}
```

The guard `i < n` is half-open: `i` takes the values `0, 1, ..., n-1`, which are exactly the valid indices of an `n`-element array. The loop runs `n` times. Change that single `<` to `<=`:

```c
for (int i = 0; i <= n; i++) {
    sum += arr[i];   /* BUG: arr[n] is one past the end */
}
```

Now `i` reaches `n`, and `arr[n]` reads the memory immediately after your array — undefined behavior in C, a buffer over-read that might return garbage, might crash, or might silently corrupt an adjacent variable. The loop ran `n + 1` times: the fence-post problem reincarnated as a comparison operator. This is why, in a half-open world, `<` is almost always correct and `<=` is the one that earns scrutiny.

The single most powerful *reading* technique for catching these — and it costs nothing, no tools, no debugger — is to mentally execute only **the first and the last iteration.** Forget the middle; the middle is always fine. Ask: on the very first iteration, what is the index, and is it valid? On the very last iteration, what is the index, and is it valid? For `i < n`: first iteration `i = 0` (valid, the first element); last iteration `i = n - 1` (valid, the last element). Both ends check out. For `i <= n`: first iteration `i = 0` (fine); last iteration `i = n` (out of bounds — caught). You just found the bug by inspecting two of the `n` iterations.

This first-and-last discipline generalizes beyond loop guards. Whenever you write code that touches a boundary, ask the two questions explicitly. A function that processes "from index `i` to index `j`": what does it do when `i == j` (the empty or single case)? When `j` is the last valid index versus one past it? A retry that runs "up to `max_retries` times": does it run `max_retries` or `max_retries + 1` times — does the initial attempt count? A sliding window of width `w` over `n` items: how many window positions are there? (It is `n - w + 1`, and the `+ 1` is a fence post; the last window starts at index `n - w`, not `n - w + 1`, and getting that wrong reads past the end on the final step.) Every one of these is the same question wearing a different costume: *what happens at the edge?*

There is a subtle variant worth flagging because it bites even careful engineers: the reverse loop. Iterating backward, the off-by-one moves to the *other* end:

```c
for (int i = n - 1; i >= 0; i--) {   /* correct: n-1 down to 0 */
    process(arr[i]);
}
```

Start at `n - 1` (the last valid index, not `n`) and stop at `>= 0` (include zero, the first element). The two classic mistakes are starting at `n` (out of bounds on the first iteration) or stopping at `> 0` (skipping element zero). Same fence-post logic, mirror-imaged. And in *unsigned* arithmetic the reverse loop has a vicious trap: if `i` is an unsigned type and you write `for (i = n; i-- > 0;)` versus a naive `for (i = n - 1; i >= 0; i--)`, the `i >= 0` is *always true* for an unsigned type — when `i` is `0` and you decrement, it wraps to a huge positive number instead of going negative, and the loop never terminates. That is a boundary bug (the value `0`) crossed with a type bug (unsigned wraparound), and it is exactly the kind of edge that only shows up when the input drives `i` all the way down to zero.

## 4. The boundary *values*: empty, single, full, and the wrap point

So far the boundaries have been about *indices* — where the loop starts and stops. The other half of this bug class is about *values*: the specific inputs whose size or shape makes the code behave differently from the typical case. These are the inputs that should be the first thing you reach for when reproducing a suspected boundary bug, because they are where the code's assumptions quietly fail. The figure below lays them out against what each one does to the code.

![A matrix mapping empty, single-element, and at-capacity inputs to what the code does and why it breaks, showing the boundary values that trigger off-by-one bugs](/imgs/blogs/off-by-one-and-boundary-bugs-3.png)

**The empty input (length zero)** is the most under-tested case in all of software, and it breaks code in two opposite ways. The first is the access that assumes at least one element: `arr[0]` on an empty array, `list.first`, `max(values)` on an empty sequence, `s[0]` on an empty string. There is no element zero, so you get an index error, a null, or in C a read of whatever happens to be at that address. The second, sneakier failure is the loop that *runs once when it should run zero times.* The classic is the `do-while`:

```c
int i = 0;
do {
    process(arr[i]);   /* runs once even when n == 0 — touches arr[0] of an empty array */
    i++;
} while (i < n);
```

A `do-while` checks its condition *after* the body, so it always executes at least once, including on empty input. A plain `while (i < n)` or `for` correctly runs zero times. Whenever you see a `do-while`, the question is reflexive: *what happens when the collection is empty?* Usually the answer is "it touches an element that does not exist."

**The single-element input (length one)** is the boundary where many divide-and-conquer algorithms quietly fall apart, because the recursion or the bisection assumes it can split the input into two non-empty halves, and a one-element input cannot be split. Binary search is the textbook casualty — we will dissect it in section seven — but the pattern is everywhere: a merge sort that recurses on `[lo, mid)` and `[mid, hi)` where `mid == lo` on a one-element range produces an empty left half and a full right half and never makes progress; a "find the median of two halves" that assumes both halves are non-empty; a string algorithm that compares `s[i]` to `s[i+1]` and walks off the end on the last character. The single-element case is the smallest input where "split into two parts" stops being possible, so it is the boundary where splitting logic reveals its bug.

**The at-capacity input (exactly the maximum size)** is where buffer boundaries turn an off-by-one into a security incident. A fixed buffer of `N` bytes has valid indices `0` through `N - 1`; writing at index `N` writes one byte past the end. Code that copies "up to `N` characters" but forgets that a C string also needs a trailing null terminator will, on an input of exactly `N` characters, write the null at index `N` — past the end. This is the entire mechanism behind a vast catalog of buffer-overflow vulnerabilities: the attacker supplies an input of *exactly* the boundary length, the off-by-one writes one byte (or a few) past the buffer, and that byte lands on a saved return address or a length field or an adjacent pointer. The fence-post error stops being a wrong number on a report and becomes remote code execution. We cover the corruption mechanics in depth in [use-after-free and memory corruption](/blog/software-development/debugging/use-after-free-and-memory-corruption); here the point is that *exactly-at-capacity* is a boundary input you must test, and the one most likely to be skipped because "who sends exactly 256 bytes?" Attackers do. Always.

**The wrap point** is the boundary that appears in arithmetic and time. An unsigned 8-bit counter wraps from `255` back to `0`; a signed 32-bit integer overflows from `2147483647` to `-2147483648`. A ring buffer's write index wraps from `capacity - 1` back to `0`, and the off-by-one is whether "full" and "empty" are distinguishable (the classic ring-buffer fix is to leave one slot empty, *because* `head == tail` is ambiguous otherwise — itself a boundary decision). Sequence numbers wrap. TCP sequence numbers wrap. Any modular arithmetic has a boundary at the modulus, and the code that is correct in the middle of the range can be wrong exactly at the wrap.

**Time boundaries** deserve their own mention because they are off-by-one bugs in disguise and they are brutal to reproduce. The last day of the month, the last *second* of the month. Midnight, where a time window that uses `<=` on the end double-counts the events at exactly `00:00:00`. Daylight-saving transitions, where the day is twenty-three or twenty-five hours long and a loop "for each hour in the day" runs the wrong number of times, or where a wall-clock time either does not exist (spring forward) or exists twice (fall back). Leap seconds and leap days. A billing window of `[month_start, month_end]` that is *closed* on both ends will, if `month_end` is `2026-06-30T00:00:00`, include the very first instant of July first when written as `<=`. The fix is the same as everywhere else: make the window half-open, `[month_start, next_month_start)`, so adjacent months tile perfectly and the boundary instant belongs to exactly one of them. For the time-ordering subtleties across machines and clocks, the database series has a thorough treatment in [time, clocks, and ordering in distributed systems](/blog/software-development/database/time-clocks-and-ordering-in-distributed-systems); the boundary lesson here is that *every* `<=` on a time endpoint is a place a double-count can hide.

#### Worked example: the empty list that crashed the dashboard

A metrics dashboard computed, for each team, the *average* request latency over the last hour. It worked beautifully in staging and for every active team in production — and threw a division error, intermittently, for a handful of teams. The on-call engineer's first instinct was "flaky infrastructure." It was not. The average was computed as `sum(latencies) / len(latencies)`, and for a team that happened to send *zero* requests in that hour, `latencies` was the empty list, `len` was `0`, and the code divided by zero.

The reason it looked intermittent was that it depended on data: a team with traffic never hit it, a team that went quiet for an hour did. The reproduction, once the hypothesis was "empty input," took thirty seconds — feed the function an empty list and watch it throw. The fix was a single guard returning `0` (or `None`, a real product decision) for the empty case. But the *prevention* was the lesson: the function had a unit test for the typical case (a list of latencies) and no test for the empty case, because the author never anticipated a team with no traffic. The empty input is the case nobody expects, which is exactly why it is the case that ships broken. After the fix, the test suite grew an explicit "empty input" test, and the team adopted a rule that any function consuming a collection gets an empty-collection test, full stop.

## 5. How to debug a boundary bug: reproduce at the edge, then shrink

When you suspect a boundary bug — and the symptoms are distinctive: a result that is off by exactly one, a crash only on certain input sizes, a duplicate or a missing element at a seam — the method is mechanical and fast. It is a tight specialization of the series' observe-reproduce-hypothesize-bisect loop, and the key move is that you *choose* the reproduction input instead of waiting for one. You do not need to capture the production input that triggered it; you can usually *construct* the triggering input from first principles, because boundary bugs trigger on a small, enumerable set of edge inputs.

**Step one: reproduce with the edge inputs first.** Before anything else, throw the canonical boundary inputs at the suspect code and watch which one breaks it. The set is short and you should keep it memorized as a checklist: empty (length zero), single element (length one), two elements (the smallest case where "first" and "last" differ), exactly at capacity (the max size), one past capacity if the API allows it, and the all-same-value input. For numeric inputs add: zero, one, negative one, the minimum and maximum of the type, and the value just below and just above any threshold the code compares against. For each of these, run the function and check the output against what you computed by hand. The input that produces a wrong answer is your reproduction, and it is far smaller and more deterministic than any production capture. The companion principle — that you have not started debugging until you can reproduce on demand — is the whole subject of [reproduce it first or you're not debugging](/blog/software-development/debugging/reproduce-it-first-or-youre-not-debugging); boundary bugs are the happy case where reproduction is *constructive*.

**Step two: shrink to the smallest failing input.** Once you have *an* input that fails, make it smaller until you have the *smallest* input that still fails. This matters enormously: a bug that reproduces on a list of one element is trivially traceable; the same bug "discovered" on a list of a million elements is a haystack. If your failing input is large, binary-search its size: does it still fail at half the size? At a quarter? Keep halving until you find the smallest `n` that triggers it. The figure below shows this narrowing, and it is the same binary-search-the-gap logic that powers `git bisect`, just applied to input size instead of commit history.

![A timeline showing the failing input size halved repeatedly from one million down to one until the single-element boundary case is isolated](/imgs/blogs/off-by-one-and-boundary-bugs-5.png)

This shrinking is so valuable that property-testing libraries automate it; when Hypothesis or QuickCheck finds a failing input, they automatically "shrink" it to the minimal counterexample before reporting, so instead of "failed on this list of 847 random integers" you get "failed on the empty list" or "failed on `[0]`." But you can shrink by hand, and on a boundary bug it converges fast because the boundaries are at the *extremes* of size — the smallest failing input is almost always empty, single, or two-element.

**Step three: trace the index at the boundary iteration.** Now that you have the smallest failing input, you can afford to watch every step. The cheapest tool is the print of the index *and the bound it is being compared against*, side by side, on every iteration:

```python
def buggy(xs):
    lo, hi = 0, len(xs)
    while lo < hi:
        mid = (lo + hi) // 2
        print(f"lo={lo} hi={hi} mid={mid}  (len={len(xs)})")   # the boundary trace
        # ... logic that updates lo or hi ...
```

Printing `lo`, `hi`, and `mid` together — *with* the length for context — turns an invisible index bug into a visible one. On a one-element input you will *see* `lo=0 hi=1 mid=0` and then watch whether the update actually narrows the range or leaves it stuck. The general technique of instrumenting with well-placed prints is covered in [logging as a debugging instrument](/blog/software-development/debugging/logging-as-a-debugging-instrument); for boundary bugs the specific discipline is *always print the index alongside the bound*, never one without the other, because the bug is precisely in their relationship.

**Step four: read the first and last iteration.** With the trace in front of you, do the reading exercise from section three explicitly. What is the index on iteration one? Is it valid? What is the index on the final iteration before the loop exits? Is *it* valid? Where exactly does the loop decide to stop, and is that one too early or one too late? Nine times out of ten the bug is visible in the first or last printed line, because that is where the boundary lives.

If a print is not enough — say the loop runs a few million times before the boundary case hits, or the bug is in compiled code where adding a print means a recompile — reach for a *conditional breakpoint*. Every real debugger lets you break only when a condition holds, which is perfect for boundary bugs: break when the index reaches the suspect value.

```bash
# gdb: stop only when the loop index hits the array length (the off-by-one moment)
$ gdb ./a.out
(gdb) break process.c:42 if i == n
(gdb) run
# execution halts exactly at the boundary iteration; now inspect i, n, and the pointer
(gdb) print i
(gdb) print n
(gdb) print arr[i]    # is this in bounds?
```

The conditional breakpoint `if i == n` does in one line what a `print` of every iteration does in a flood of output: it stops you *exactly* at the boundary moment and nowhere else. Mastering conditional and data breakpoints is the subject of [the debugger is a microscope, use it](/blog/software-development/debugging/the-debugger-is-a-microscope-use-it); the boundary-specific trick is to set the condition to the *exact edge value* — `i == n`, `lo == hi`, `offset == total` — so you skip the millions of fine iterations and land on the one that matters.

**Step five: binary-search which input size flips the behavior.** If you cannot tell *whether* a bug is size-dependent, make it tell you. Run the function over a sweep of input sizes — `0, 1, 2, 3, ..., 16, 32, ...` — and record which sizes pass and which fail. A boundary bug produces a sharp, structured pattern: fails on `0` and `1`, passes from `2` up; or passes everywhere except exact powers of two; or fails only at the exact capacity. That pattern *is* the diagnosis. A bug that fails on `0` and `1` but passes from `2` is screaming "I cannot handle inputs too small to split." A bug that fails only at one specific size is screaming "I have an exact-capacity boundary." You read the root cause directly off the pass/fail-versus-size table.

## 6. The taxonomy: sorting a boundary bug to find its fix

Once you have reproduced and traced, it helps to *classify* the bug, because the class tells you the fix. Boundary bugs split cleanly into two families, and knowing which one you have points straight at the remedy. The figure below is that taxonomy.

![A decision tree splitting boundary bugs into index-arithmetic mistakes and boundary-value mistakes, each branching into specific subtypes with their fixes](/imgs/blogs/off-by-one-and-boundary-bugs-6.png)

The first family is **index-arithmetic mistakes** — the bug is in *where* you read or write, the computed position is off by one. This splits into the *fence-post* subtype (you used `n` where you needed `n - 1`, or computed a count of items where you needed a count of gaps) and the *loop-guard* subtype (`<` versus `<=`, the wrong start or stop). The fix for index-arithmetic bugs is almost always the same: adopt half-open consistently, derive the bound from the convention rather than guessing it, and prefer high-level iteration that hides the index entirely (`for x in xs`, `enumerate`, iterators) so there is no arithmetic to get wrong. When you must compute an index, sanity-check it against the convention: the last valid index of an `n`-element array is `n - 1`, the number of windows of width `w` is `n - w + 1`, the number of pairs is `n * (n - 1) / 2` — memorize these and check your arithmetic against them.

The second family is **boundary-value mistakes** — the indices are fine, but the code does the wrong thing for a particular *input*. This splits into the *empty-or-single* subtype (length zero or one breaks an assumption that there is at least one element, or at least two) and the *capacity-or-wrap* subtype (the max size, the wrap point, midnight, the DST transition). The fix for boundary-value bugs is to *handle the edge explicitly* — an early return for the empty case, a base case in the recursion for the single element, a check against the maximum before writing — and then *test every one of those edges*. You cannot fix a boundary-value bug by being more careful with arithmetic; the arithmetic was fine. You fix it by enumerating the special inputs and giving each one correct behavior.

This classification is not academic. When a boundary bug lands on your desk, asking "is this index arithmetic or a boundary value?" immediately halves the search space. If the same code gives a *wrong-by-one* answer on *typical* input, it is index arithmetic — go read the loop guards and the index computations. If the code is *correct on typical input* but *crashes or misbehaves on a special input* (empty, single, max), it is a boundary value — go enumerate the special cases and check each one. The symptom tells you the family; the family tells you where to look.

| Symptom | Likely family | Where to look first | Confirming test |
| --- | --- | --- | --- |
| Result off by exactly one on normal input | Index arithmetic (fence-post) | Index computations, `+ 1` / `- 1`, count vs gap | Hand-compute the expected count |
| Reads/writes one past the end | Index arithmetic (loop guard) | The `<` vs `<=` in the guard, start/stop | First and last iteration index |
| Crashes only on empty or tiny input | Boundary value (empty/single) | The `do-while`, the `arr[0]`, the split logic | Run with empty and single inputs |
| Misbehaves only at max size or a wrap | Boundary value (capacity/wrap) | The buffer write at capacity, modular math | Run with exactly-max and max+1 |
| Duplicate or missing row at a seam | Index arithmetic (interval mix) | The `<=` vs `<` on the range boundary | Assert union of ranges = full set |
| Double-count at midnight / month-end | Boundary value (time) | The `<=` on the time-window end | Half-open the window, recount |

## 7. The classic: a binary search that hangs on one element

No post about boundary bugs is honest without the canonical example, because it is canonical for a reason: binary search is *simple* — fewer than ten lines — and yet a famous study found that a huge fraction of published implementations were broken, and most of the breakage was at the boundaries. It is the perfect specimen: the bug lives entirely at the single-element and empty edges, it is invisible on typical input, and you debug it by tracing the index at exactly those edges.

Here is a binary search with the classic boundary bug. Read it and try to spot where it goes wrong on a one-element array before reading on:

```python
def binary_search_buggy(xs, target):
    lo, hi = 0, len(xs)          # half-open: hi is one PAST the last index
    while lo <= hi:              # BUG 1: should be lo < hi for a half-open range
        mid = (lo + hi) // 2
        if xs[mid] == target:    # BUG 2 lurking: mid can equal len(xs), out of bounds
            return mid
        elif xs[mid] < target:
            lo = mid + 1
        else:
            hi = mid             # BUG 3: with hi exclusive, this can fail to narrow
    return -1
```

There are two conventions tangled together here, and that tangle *is* the bug. The code initializes `hi = len(xs)`, which is the *half-open* convention — `hi` is one past the last valid index. But then the loop guard is `lo <= hi`, the *closed* convention. And the `hi = mid` update is half-open style, while a closed search would use `hi = mid - 1`. The mismatch produces two distinct failures. On the *empty* array, `lo = 0`, `hi = 0`, the guard `0 <= 0` is true, so the loop enters, computes `mid = 0`, and evaluates `xs[0]` — an index error on an array with no elements. On a *single-element* array `[5]` searching for something not present, the range can collapse to `lo == hi` and, because the guard is `<=` instead of `<`, the loop runs *again* on the empty range, recomputes the same `mid`, and either re-checks the same element forever or indexes out of bounds. The figure below traces the index walk on the one-element case for both the buggy and the fixed update, and the difference is the entire fix.

![A grid tracing the binary search indices lo, mid, and hi on a single-element array, contrasting the buggy update that never narrows with the fix that halts](/imgs/blogs/off-by-one-and-boundary-bugs-7.png)

The fix is to commit *fully* to one convention. Going all-in on half-open:

```python
def binary_search(xs, target):
    lo, hi = 0, len(xs)          # half-open: search the range [lo, hi)
    while lo < hi:               # strictly less: empty range [k, k) exits immediately
        mid = lo + (hi - lo) // 2  # also avoids lo+hi overflow in fixed-width ints
        if xs[mid] == target:
            return mid
        elif xs[mid] < target:
            lo = mid + 1         # narrow to [mid+1, hi)
        else:
            hi = mid             # narrow to [lo, mid)
    return -1                    # lo == hi means empty range, not found
```

Trace it on the boundaries. Empty array: `lo = 0`, `hi = 0`, guard `0 < 0` is false, loop never runs, returns `-1`. Correct, and no out-of-bounds access. Single-element array `[5]` searching for `7`: `lo = 0`, `hi = 1`, guard `0 < 1` true, `mid = 0`, `xs[0] = 5 < 7`, so `lo = mid + 1 = 1`. Now `lo = 1`, `hi = 1`, guard `1 < 1` false, exit, return `-1`. Correct. The single-element case *makes progress* now because the half-open guard `lo < hi` treats `lo == hi` as the empty range and stops, and the `lo = mid + 1` update strictly advances `lo` past `mid` so the range always shrinks. The buggy `<=` version never shrank to empty; it shrank to `lo == hi` and then ran one fatal extra time.

Notice the second, sneakier fix folded into the corrected version: `mid = lo + (hi - lo) // 2` instead of `mid = (lo + hi) // 2`. In a language with fixed-width integers — C, Java, Go, Rust — `lo + hi` can *overflow* when both are large, wrapping to a negative number and producing a negative or wrong `mid`. This is itself a boundary bug, at the *value* boundary of the integer type, and it is exactly the bug that the famous "nearly all binary searches are broken" article was about: a binary search that is correct for small arrays and silently wrong for arrays larger than about half the maximum integer. Computing the midpoint as `lo + (hi - lo) / 2` keeps the intermediate values in range because `hi - lo` is always smaller than `hi`. Two boundary bugs in one ten-line function: the empty/single index boundary and the integer-overflow value boundary. That density is *why* binary search is the canonical example.

#### Worked example: the search that hung 1 time in 100,000

A search service had a binary search over a sorted in-memory index, and once every so often — roughly one request in a hundred thousand — a worker thread would peg a CPU core at one hundred percent and stop responding. Restarting the worker cleared it; the request that triggered it was never logged because the worker hung before logging. The team chased it as a "thread starvation" or "GC pause" problem for two days, because the symptom (one stuck thread, high CPU) looked like a runtime issue, not a logic bug.

The breakthrough was reproducing at the boundary. Someone hypothesized "what if the index is sometimes a single element?" and wrote a test that called the search on arrays of size `0` and `1` for a target that was *not present*. The size-one, not-present case hung instantly — a `while lo <= hi` that never narrowed, spinning forever, which is exactly the one-hundred-percent-CPU symptom. The reason it was one-in-a-hundred-thousand in production was that the index was almost always large; only when a particular shard had been filtered down to a single matching candidate did the search run on a one-element array, and only when the target was absent did it hit the infinite loop. The bug rate was just the rate at which that data shape occurred.

The fix was the one-character change from `<=` to `<` plus committing to half-open. The *proof* was a property test (next section) that generated arrays of every size from `0` to `1000` and every target including absent ones, ran the search ten thousand times, and asserted it always *terminated* and always returned the correct index or `-1`. Before the fix: hung on size-one absent-target inputs, which the random generator hit within the first few hundred cases. After the fix: ten thousand cases, zero hangs, zero wrong answers, and a wrapped timeout assertion to guarantee termination was actually being tested. Two days of "thread starvation" investigation collapsed into a thirty-second reproduction once the hypothesis was "boundary input," because the boundary input was constructible from first principles instead of waiting for the one-in-a-hundred-thousand production request.

## 8. Preventing it forever: property tests that hammer the edges

Finding a boundary bug once is satisfying. Making the entire class impossible to reintroduce is the real win, and it comes down to one shift in how you test: stop hand-picking a few example inputs and start *generating* inputs that deliberately include the edges. Example-based tests fail at boundaries for a structural reason — humans write tests for the cases they imagine, and nobody imagines the empty list, the single element, the exactly-at-capacity input, the off-by-one inversion. Those are precisely the cases the bug lives in. The figure below contrasts the two approaches.

![A before-and-after comparison of hand-written example tests that skip the edges versus property-based tests that shrink to the empty, single, and max inputs](/imgs/blogs/off-by-one-and-boundary-bugs-8.png)

**Property-based testing** flips the model. Instead of "for *this* input I expect *that* output," you state a *property* that must hold for *all* inputs — and the library generates hundreds of inputs, deliberately biased toward the edges, trying to falsify it. Python's Hypothesis, Haskell's and Scala's QuickCheck, JavaScript's fast-check, Rust's proptest all work this way, and crucially they *automatically include* the boundary values: the empty collection, the single element, zero, the integer min and max, the empty string. They are *built* to find off-by-one bugs, because off-by-one bugs are exactly the bugs that hide from example tests and surface under generated edges. Here is a property test that would have caught the binary-search hang on its first run:

```python
from hypothesis import given, strategies as st, settings

@given(xs=st.lists(st.integers()).map(sorted), target=st.integers())
@settings(deadline=200)   # fail if any single case takes > 200ms — catches the infinite loop
def test_binary_search_matches_linear(xs, target):
    # Property 1: the result must agree with a trivially-correct linear scan.
    expected = xs.index(target) if target in xs else -1
    result = binary_search(xs, target)
    if expected == -1:
        assert result == -1
    else:
        assert result != -1 and xs[result] == target
    # Property 2 (invariant): it must terminate — guaranteed by the deadline above.
```

Hypothesis will generate the empty list, single-element lists, lists where `target` is and is not present, and targets at the extreme ends — automatically, on the first run, biased toward exactly the edges. The `deadline=200` turns the infinite-loop bug into a *failure* rather than a hang, so the buggy `<=` version fails loudly instead of spinning forever. And when it finds a failing case, it *shrinks* it to the minimal counterexample — almost certainly `xs=[]` or `xs=[0]` — so the report names the boundary directly. That shrinking is the same input-minimization you would do by hand in section five, automated.

The pagination bug from section two yields to the same weapon, and the property is beautiful because it is exactly the *tiling* property of half-open intervals:

```python
from hypothesis import given, strategies as st

@given(total=st.integers(min_value=0, max_value=500),
       page_size=st.integers(min_value=1, max_value=50))
def test_pages_tile_the_full_set(total, page_size):
    full = list(range(1, total + 1))      # ids 1..total
    seen = []
    last_seen = 0
    while True:
        page = fetch_page(full, last_seen, page_size)   # the function under test
        if not page:
            break
        seen.extend(page)
        last_seen = page[-1]
    # THE property: every page concatenated must equal the full set, no dups, no gaps.
    assert seen == full, f"pages did not tile: got {len(seen)} ids, expected {len(full)}"
```

The single assertion `seen == full` catches *both* failure modes of the off-by-one: a duplicated row makes `seen` longer than `full`, a skipped row makes it shorter or out of order. With `total` swept across zero (the empty case), one, and values that are *exact multiples* of `page_size` (the seam case where `total // page_size` is exact and the last page is full), Hypothesis drives the test straight at the page boundaries. The closed-guard version (`id <= last_seen + page_size` combined with an inclusive low end) duplicates a row at every seam and fails this property on the first multiple-of-page-size input it generates; the half-open version passes for every `total` and `page_size`. This is the property test mentioned in the scope as the thing that catches the pagination bug, and it is worth internalizing the shape: *assert that the union of the pages equals the full set.* That single invariant is a complete specification of correct pagination.

Beyond property testing, four more habits shrink this bug class toward zero:

**Explicit boundary unit tests — the empty/one/two/max table.** Even without a property-testing library, *deliberately* write a test for each boundary: empty input, single element, two elements, exactly-at-capacity. Make it a table-driven test so adding a new function means filling in the same row of edges. The discipline is to never consider a collection-consuming function "tested" until its empty, single, and max cases each have an explicit assertion. Most off-by-one bugs that reach production would have died in code review against a checklist of four cases.

**Invariant assertions.** Assert the things that must always be true, right in the code, so a boundary violation fails *loudly and immediately* instead of silently corrupting data. `assert lo <= hi` at the top of a binary-search loop catches an inverted range the instant it happens. `assert 0 <= idx < len(arr)` before an index catches an out-of-bounds before it reads garbage. `assert sum_of_pages == total_rows` after pagination catches a tiling violation. The assertion turns an invisible off-by-one into a stack trace pointing at the exact line, which is the difference between a five-minute fix and a five-day investigation.

**Choose half-open and hold it.** This is the cheapest prevention of all and it is a *cultural* choice as much as a technical one: agree, as a codebase, that ranges are half-open `[start, end)` unless there is a screaming reason otherwise, and write a lint rule or a code-review reflex around `<=` in loop guards and `+ 1` in index math. Consistency is what makes the *tiling* property hold across module boundaries; the bugs come from the seams where one module's closed convention meets another's half-open one.

**Prefer high-level iteration over manual indices.** The off-by-one you cannot write is the one you cannot reintroduce. `for x in xs` has no index to get wrong. `for i, x in enumerate(xs)` gives you the index *and* the element with no arithmetic. `itertools.pairwise(xs)`, `zip(xs, xs[1:])`, slicing, comprehensions — every one of these replaces manual `arr[i]` indexing with a construct that handles the boundary for you. When you find yourself writing `for i in range(len(xs))` and then using `xs[i]`, stop: that is the highest-risk form, and there is almost always a boundary-safe iterator that expresses the same thing.

| Prevention technique | Catches | Cost | When to reach for it |
| --- | --- | --- | --- |
| Property-based test (Hypothesis/QuickCheck) | Empty, single, max, inversions — automatically | Medium (learn the library) | Any function with a clear correctness invariant |
| Explicit boundary unit tests | The specific edges you enumerate | Low | Every collection-consuming function |
| Invariant assertions in code | Boundary violations at runtime, loudly | Low | Loops, index math, range merges |
| Half-open convention + lint | Convention-mixing at the seams | Low (one-time) | Codebase-wide policy |
| High-level iteration (no manual index) | The off-by-one you never write | Negative (less code) | Whenever you would write `for i in range(len)` |
| Fuzzing | Crashes on capacity/overflow boundaries | High (setup + triage) | Parsers, buffers, untrusted input |

## 9. War story: real boundary bugs that mattered

Boundary bugs are not academic curiosities. Some of the most consequential software failures in history were off-by-one or boundary errors, and it is worth knowing them because they calibrate *how seriously* to take the edge.

**Heartbleed (2014)** is the most famous buffer-boundary bug of the modern era, and it is a boundary-value bug to its core. OpenSSL's implementation of the TLS heartbeat extension let a client say "here is a payload of length `N`, echo it back." The bug: the code trusted the client-supplied length `N` without checking it against the *actual* length of the payload the client sent. An attacker sent a one-byte payload but claimed `N` was sixty-four kilobytes. OpenSSL's `memcpy` then copied sixty-four kilobytes starting at the payload — reading far past the boundary of the real payload into adjacent server memory, which often contained private keys, passwords, and session tokens — and sent it all back to the attacker. The fix was a single bounds check: verify that the claimed length does not exceed the actual record length. One missing comparison against a boundary, and a substantial fraction of the internet's secrets were readable by anyone. The lesson is the exactly-at-capacity-and-beyond input: when a length comes from outside, the boundary check is not optional, and the attacker *will* send the value that is one — or sixty-four thousand — past the end.

**The Knight Capital deploy (2012)** was a different flavor — a deployment boundary, an off-by-one in *which servers got the new code.* Knight Capital deployed new trading software to eight servers, but one of the eight did not get the update due to a manual deployment process, and that one server still ran old code that repurposed a flag the new code used for something else. When the flag was set, the stale server began sending erroneous orders into the market at high speed. In forty-five minutes the firm lost roughly four hundred forty million dollars and was effectively destroyed. The boundary here was "all eight" versus "seven of eight" — a fence-post in deployment, the one server that fell off the end of the count. It is a reminder that boundary bugs are not only in loops; "did *every* node get the update?" is the same `n` versus `n - 1` question, and the missing one is where the disaster lives. The microservices and deployment-safety angle is exactly why automated, all-or-nothing rollouts exist; the broader lesson on safe deploys lives in the system-design literature on outages.

**The Therac-25 (1980s)** medical radiation machine killed and maimed patients partly through a *counter overflow* boundary bug. A one-byte counter that incremented on each pass was supposed to ensure a safety check ran; when it overflowed from `255` back to `0` at exactly the wrong moment, the check was skipped and the machine could deliver a massive radiation overdose. The boundary was the wrap point of an eight-bit counter — value `255` rolling to `0` — combined with a race condition on operator input. It is the canonical case study in software safety, and the boundary that killed people was a single byte wrapping around. When the cost of a boundary bug is measured in human lives, "test the wrap point" stops being pedantry.

**The Year 2038 problem** is a boundary bug scheduled for the future: on January 19, 2038, a signed 32-bit Unix timestamp — seconds since 1970 — will overflow from its maximum positive value and wrap to a large negative number, suddenly representing a date in 1901. Any system still using 32-bit `time_t` will compute wildly wrong dates, and the failures will look exactly like the integer-overflow boundary in binary search, just on a clock instead of a midpoint. It is the same mechanism as the famous Year 2000 problem (two-digit years wrapping from `99` to `00`) — a value boundary in time representation. These are off-by-one's big cousins: not off by one *element*, but off by one *past the representable range*, which is the same fence post at the edge of a type.

The thread through all of these is that the boundary is where the *assumption* lives, and the assumption is what the unusual input violates. Heartbleed assumed the claimed length matched the real length. Knight assumed all eight servers updated. Therac assumed the counter would not wrap at that instant. Y2038 assumes seconds-since-1970 fits in 31 bits. Every one is correct in the middle of the range and catastrophic at the edge — which is the whole thesis of this post, written in the largest possible letters.

## 10. Stress-testing your fix: what if it only breaks under load, in release, on one host?

Finding and fixing a boundary bug in a clean reproduction is the easy version. The hard version is when the boundary only manifests under conditions you cannot easily recreate, and a senior engineer's value is largely in handling those. Let me walk through the stress cases, because each one has a specific technique.

**What if it only reproduces under load?** Some boundary bugs need a *specific data shape* that only occurs at scale — the shard that filters down to exactly one element, the queue that hits exactly capacity, the buffer that fills to its exact limit. The fix is not to reproduce the load; it is to reproduce the *shape*. Reason backward from the boundary to the input that produces it, then construct that input directly. The binary-search hang needed a one-element array; you do not need production traffic to make a one-element array. If you genuinely cannot guess the shape, instrument the boundary in production with a cheap counter — log whenever the array reaches size one, whenever the buffer reaches capacity — and let production tell you which boundary it is hitting. That is the observability move: you cannot attach a debugger to the payments process in prod, but you can add a metric that increments at the boundary, and [observability for debugging prod](/blog/software-development/debugging/observability-for-debugging-prod) covers how to do that without a redeploy storm.

**What if it only breaks in release builds, not debug?** This is the heisenbug variant, and for boundary bugs it usually means *undefined behavior* — a read or write one past the end that, in a debug build, happens to land on padding or a guard region and does nothing visible, but in an optimized release build lands on a live variable or gets the compiler's optimizer to assume the out-of-bounds access never happens and "optimize away" your bounds check. The fix is not to debug the release build by hand; it is to run a sanitizer that makes the boundary violation *deterministic*. AddressSanitizer instruments every memory access and traps the *instant* you read or write past an allocation, in any build:

```bash
# Compile with AddressSanitizer; it traps the moment of the off-by-one, with a stack trace.
$ gcc -fsanitize=address -g -O1 buggy.c -o buggy
$ ./buggy
# =================================================================
# ==12345==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x...
#   WRITE of size 1 at 0x... thread T0
#     #0 0x... in copy_string buggy.c:18    <-- the exact off-by-one line
#   0x... is located 0 bytes to the right of 256-byte region
```

The phrase "0 bytes to the right of 256-byte region" is the signature of the at-capacity boundary bug: you wrote exactly one byte past a 256-byte buffer. ASan turns a release-only heisenbug into a deterministic crash *at the boundary access*, with the line number. For the full mechanics of how the corruption manifests far from its cause, see [use-after-free and memory corruption](/blog/software-development/debugging/use-after-free-and-memory-corruption); the boundary point is that *you do not debug an off-by-one by staring at a release build — you compile with a sanitizer that traps at the edge.*

**What if it only happens on one host?** Boundary bugs that are host-specific usually mean the boundary depends on something environmental: a different timezone (the midnight and DST boundaries shift), a different locale (string-length boundaries shift with multi-byte characters), a different integer width (32-bit versus 64-bit changes the overflow boundary), a different page size or alignment. The technique is to *diff the environment*: what is different about the one host? Timezone, locale, word size, library versions. A billing bug that only happens on the host configured for a timezone with DST is a time-boundary bug; a string bug that only happens on the host with a UTF-8 locale is a multi-byte-boundary bug. The host is not random — it is the one whose environment pushes the input across a boundary the others do not reach.

**What if it only shows up after hours of running?** A boundary that is reached only after long accumulation — a counter that overflows after four billion increments, a timestamp that wraps, a sequence number that rolls over, a buffer index that wraps after enough writes — will not show in a short test. The technique is to *fast-forward to the boundary*: seed the counter near its maximum, set the clock near the wrap, pre-fill the buffer near capacity, and *then* run the test. You do not wait four billion increments; you start at four billion minus ten and watch the wrap happen in ten steps. The Therac counter overflowed at 256; you test by starting at 250.

**What if you cannot attach a debugger at all?** In a locked-down production environment, or a process you must not pause (the payments service, the trading engine), reproduction must move *out* of prod entirely. Capture the boundary input — the one-element array, the at-capacity payload — via a log or a metric, then reproduce it offline where you *can* attach a debugger and run a sanitizer. The whole point of shrinking to the smallest failing input is that the smallest input is *portable*: a one-element array reproduces anywhere, so you never need to debug at the boundary in production, only to *learn the boundary's shape* from production and recreate it on your laptop.

## 11. How to reach for this (and when not to)

Every technique in this post has a cost, and a principal engineer's job is partly knowing when *not* to deploy the heavy machinery. Here is the decisive version.

**Always do the first-and-last-iteration read.** It is free, it takes ten seconds, and it catches the majority of off-by-one bugs before they exist. Whenever you write a loop or compute an index, mentally execute iteration one and the final iteration and check both indices are valid. There is no situation where skipping this is correct; it is the cheapest, highest-yield habit in the entire post.

**Always test the empty case for collection-consuming functions.** If a function takes a list, a string, a map, an array — write the empty-input test. It is the single most under-tested case and the single most common production boundary bug. The cost is one test; the benefit is never shipping a divide-by-zero on the quiet team, a crash on the empty result set, a `do-while` that runs once on no data.

**Reach for property-based testing when the function has a clear invariant** — a sort produces ordered output, pagination tiles to the full set, a search agrees with a linear scan, an encode-then-decode round-trips to the original. When you can state a property over *all* inputs, Hypothesis or QuickCheck will hammer the edges far harder than you can by hand, and the automatic shrinking hands you the minimal boundary counterexample for free. The cost is learning the library and writing the property; for any function with a crisp correctness condition it pays for itself the first time it shrinks a failure to the empty list.

**Reach for a sanitizer (ASan/UBSan) when the boundary bug is a memory or undefined-behavior boundary** — an out-of-bounds read or write, an integer overflow, a buffer at capacity — especially when it is release-only or host-specific. ASan is the right tool when "off by one" means "one byte past the allocation"; it traps at the boundary access with a line number. Reach for a conditional breakpoint (`break ... if i == n`) when you need to land on the exact boundary iteration in a loop that runs millions of times.

Now the *don'ts.* **Don't reach for property-based testing when the correctness condition is not a clean invariant.** If the only way to state the expected output is to reimplement the function, the property test is just a second copy of the bug. Some functions are better served by a handful of carefully chosen boundary examples than by generated inputs with no checkable property. **Don't run a sanitizer in production** — ASan roughly doubles memory and adds significant CPU overhead; it is a development and CI tool, not a prod tool. **Don't add a conditional breakpoint when a print of `i` and the bound answers the question** — the print is faster to add and gives you the whole sequence at once; reach for the debugger only when the loop is too long to flood with output or you are in compiled code where a print means a recompile. **Don't chase a boundary heisenbug in an optimized build** when you have not first tried it under a sanitizer or at `-O0`; the optimizer's assumptions about undefined behavior will lie to you. And **don't over-engineer the common path to guard against a boundary that cannot occur** — if a list is provably non-empty by construction (you just appended to it), an empty-case guard is dead code that obscures intent. The skill is calibrating the defense to the actual boundary, not bolting a check onto every line.

A note on where boundary bugs sit relative to the rest of this series: they overlap heavily with the *null/empty* family ([the null, the undefined, and the empty](/blog/software-development/debugging/the-null-the-undefined-and-the-empty)), because empty is both a null-shaped problem and a boundary-shaped one; with *memory corruption*, because the at-capacity write is the gateway to the buffer overflow; and with *bisection* ([binary-search your bug with bisection](/blog/software-development/debugging/binary-search-your-bug-with-bisection)), because shrinking the failing input is the same binary-search-the-gap move applied to input size. The boundary specialization is the *constructive* reproduction — you can build the triggering input from the convention rather than waiting for production to hand it to you — and that is what makes this bug class, for all its notoriety, one of the most *tractable* to debug once you know to look at the edge first.

## 12. Key takeaways

- **The edge is where the bug lives; the middle is always fine.** Off-by-one and boundary bugs cluster at empty, single, full, the last element, the wrap point, and the seam, because that is where two ways of counting collide. When you suspect one, test those inputs *first* — they are constructible from first principles, no production capture needed.
- **Most off-by-one bugs are a fence-post error in disguise:** a count of *items* used where a count of *gaps* was needed, or vice versa. When a result is wrong by exactly one, that mismatch is the first hypothesis to check.
- **Pick half-open `[start, end)` and hold it everywhere.** Its length is `end - start` with no fence post, adjacent ranges tile with no overlap or gap, and the empty range is just `[k, k)`. The bugs come from the seams where a closed convention leaks into a half-open world — so treat every `<=` in a loop guard and every `+ 1` in index math as a suspect.
- **Read the first and last iteration, always.** It is free and it catches most index bugs before they ship: on iteration one and the final iteration, is the index valid? Where exactly does the loop stop, and is that one too early or one too late?
- **Shrink the failing input to the smallest case that still breaks.** A bug on a one-element array is trivially traceable; the same bug on a million elements is a haystack. Binary-search the input *size* until the boundary case is naked.
- **Trace the index alongside the bound.** Print `lo`, `hi`, and `mid` together — never one without the other — because the bug is in their relationship. Use a conditional breakpoint (`if i == n`) to land on the exact boundary iteration.
- **Property-based testing is the off-by-one assassin.** Hypothesis, QuickCheck, and friends generate the empty, single, and max inputs you would never hand-pick, and shrink any failure to the minimal counterexample. Assert real invariants: the union of pages equals the full set; the search agrees with a linear scan; encode-then-decode round-trips.
- **At-capacity is a security boundary.** The write one byte past a full buffer is the mechanism behind a vast catalog of overflow vulnerabilities. When a length comes from outside, the bounds check is not optional — the attacker will send the value that is one past the end.
- **Compile with a sanitizer for memory-boundary bugs.** AddressSanitizer turns a release-only off-by-one into a deterministic crash at the exact offending access, with a line number, and tells you "0 bytes to the right of an N-byte region" — the signature of the at-capacity write.
- **Time boundaries are off-by-one in disguise.** Midnight, month-end, DST, and the wrap point double-count or skip whenever a window uses `<=` on its end. Make every time window half-open `[start, next_start)` so adjacent periods tile and the boundary instant belongs to exactly one.

## Further reading

- [Stop guessing: the scientific method of debugging](/blog/software-development/debugging/stop-guessing-the-scientific-method-of-debugging) — the observe-reproduce-hypothesize-bisect-fix-prevent loop this post specializes for boundary bugs.
- [Reproduce it first, or you're not debugging](/blog/software-development/debugging/reproduce-it-first-or-youre-not-debugging) — why a deterministic reproduction comes before everything, and why boundary bugs are the happy case where you can *construct* it.
- [Use-after-free and memory corruption](/blog/software-development/debugging/use-after-free-and-memory-corruption) — how an at-capacity write past a buffer corrupts the heap and crashes far from its cause, and how sanitizers catch it.
- [Binary-search your bug with bisection](/blog/software-development/debugging/binary-search-your-bug-with-bisection) — the same halve-the-gap logic this post applies to input size, applied to commit history.
- [The null, the undefined, and the empty](/blog/software-development/debugging/the-null-the-undefined-and-the-empty) — the sibling bug class where empty meets null, and how the empty boundary breaks code two opposite ways.
- "Nearly All Binary Searches and Mergesorts are Broken" (Joshua Bloch, Google Research Blog, 2006) — the canonical write-up of the `(lo + hi) / 2` integer-overflow boundary in binary search.
- *Why Programs Fail: A Guide to Systematic Debugging* by Andreas Zeller — the rigorous treatment of input minimization and delta debugging that automates the "shrink to the smallest failing input" move.
- The Hypothesis documentation (hypothesis.readthedocs.io) and the original QuickCheck paper (Claessen and Hughes, 2000) — property-based testing and the automatic shrinking that hands you the minimal boundary counterexample.
